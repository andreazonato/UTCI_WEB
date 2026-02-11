#!/usr/bin/env python3
import argparse
import json
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

NODATA_FLOOR = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raster click server for timeseries queries.")
    parser.add_argument(
        "--solweig-dir",
        type=str,
        required=True,
        help="Base directory containing solweig_gpu scenario folders.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the server.",
    )
    return parser.parse_args()


@lru_cache(maxsize=8)
def open_dataset(path: str):
    return rasterio.open(path)


@lru_cache(maxsize=4096)
def resolve_tif_path(solweig_dir: str, folder: str, tif_name: str) -> str | None:
    """
    Resolve a tif path robustly:
    1) exact solweig_dir/folder/tif_name
    2) exact solweig_dir/tif_name
    3) unique recursive match under solweig_dir (fallback)
    """
    base = Path(solweig_dir).resolve()
    folder = (folder or "").strip()
    candidates: list[Path] = []

    if folder:
        candidates.append((base / folder / tif_name).resolve())
    candidates.append((base / tif_name).resolve())
    for cand in candidates:
        try:
            if cand.exists() and cand.is_file():
                return str(cand)
        except Exception:
            continue

    # Fallback for portable deployments where folder in map might be ".".
    matches = list(base.rglob(tif_name))
    if len(matches) == 1:
        return str(matches[0].resolve())
    return None


def _xy_from_lonlat(ds, lon: float, lat: float) -> tuple[float, float]:
    if ds.crs is None:
        raise ValueError("Dataset CRS is missing.")
    if ds.crs.to_epsg() != 4326:
        xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
        return xs[0], ys[0]
    return lon, lat


def _valid_value(v: float | None, nodata: float | None) -> bool:
    if v is None:
        return False
    try:
        vf = float(v)
    except Exception:
        return False
    if nodata is not None and vf == nodata:
        return False
    if not (vf == vf):
        return False
    if vf <= NODATA_FLOOR:
        return False
    return True


def _sanitize_series(values, nodata: float | None) -> list:
    out = []
    for v in values:
        out.append(float(v) if _valid_value(v, nodata) else None)
    return out


def _sample_all_bands_xy(ds, x: float, y: float) -> list:
    vals = next(ds.sample([(x, y)], indexes=list(range(1, ds.count + 1))))
    return _sanitize_series(vals, ds.nodata)


def _find_nearest_valid_xy(ds, x: float, y: float, max_radius: int = 8) -> tuple[float, float] | None:
    try:
        row0, col0 = ds.index(x, y)
    except Exception:
        return None
    if ds.width <= 0 or ds.height <= 0:
        return None
    row0 = max(0, min(ds.height - 1, int(row0)))
    col0 = max(0, min(ds.width - 1, int(col0)))

    nodata = ds.nodata
    for r in range(1, max_radius + 1):
        r0 = max(0, row0 - r)
        r1 = min(ds.height - 1, row0 + r)
        c0 = max(0, col0 - r)
        c1 = min(ds.width - 1, col0 + r)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h <= 0 or w <= 0:
            continue
        arr = ds.read(1, window=Window(c0, r0, w, h))
        if arr.size == 0:
            continue
        mask = np.isfinite(arr) & (arr > NODATA_FLOOR)
        if nodata is not None:
            mask &= arr != nodata
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        # Pick closest valid pixel to the clicked location.
        dy = ys - (row0 - r0)
        dx = xs - (col0 - c0)
        idx = int(np.argmin(dx * dx + dy * dy))
        rr = int(r0 + ys[idx])
        cc = int(c0 + xs[idx])
        xx, yy = ds.xy(rr, cc)
        return float(xx), float(yy)
    return None


def sample_timeseries(ds, lon: float, lat: float, *, allow_nearest: bool = True) -> tuple[list, bool]:
    x, y = _xy_from_lonlat(ds, lon, lat)
    series = _sample_all_bands_xy(ds, x, y)
    if any(v is not None for v in series):
        return series, False
    if not allow_nearest:
        return series, False
    near = _find_nearest_valid_xy(ds, x, y)
    if near is None:
        return series, False
    return _sample_all_bands_xy(ds, near[0], near[1]), True


def sample_value(ds, lon: float, lat: float, *, allow_nearest: bool = True) -> float | None:
    values, _ = sample_timeseries(ds, lon, lat, allow_nearest=allow_nearest)
    return values[0] if values else None


def sample_mean_value(ds, lon: float, lat: float, *, allow_nearest: bool = True) -> float | None:
    values, _ = sample_timeseries(ds, lon, lat, allow_nearest=allow_nearest)
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/timeseries":
            if parsed.path != "/value":
                self.send_response(404)
                self.end_headers()
                return

        qs = parse_qs(parsed.query)
        folder = qs.get("folder", [""])[0]
        var = qs.get("var", [""])[0]
        date = qs.get("date", [""])[0]
        lat = qs.get("lat", [""])[0]
        lng = qs.get("lng", [""])[0]
        tif = qs.get("tif", [""])[0]
        mean = qs.get("mean", [""])[0]

        if not folder or not var or not date or not lat or not lng:
            if parsed.path == "/timeseries":
                self.send_response(400)
                self.end_headers()
                return

        try:
            lat_f = float(lat)
            lng_f = float(lng)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return

        if parsed.path == "/timeseries":
            tif_name = f"{var}_{date}.tif"
        else:
            if mean == "1":
                tif_name = f"{var}_{date}.tif"
            else:
                tif_name = tif
        tif_path = resolve_tif_path(self.server.solweig_dir, folder, tif_name)
        if not tif_path:
            self.send_response(404)
            self.end_headers()
            return

        try:
            ds = open_dataset(tif_path)
            if parsed.path == "/timeseries":
                values, used_nearest = sample_timeseries(ds, lng_f, lat_f, allow_nearest=True)
                payload = {"values": values, "used_nearest": used_nearest}
            else:
                if mean == "1":
                    value = sample_mean_value(ds, lng_f, lat_f, allow_nearest=True)
                else:
                    value = sample_value(ds, lng_f, lat_f, allow_nearest=True)
                payload = {"value": value}
        except Exception:
            self.send_response(500)
            self.end_headers()
            return

        payload = json.dumps(payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))


def main() -> None:
    args = parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    server.solweig_dir = args.solweig_dir
    print(f"Serving raster click API on http://{args.host}:{args.port}")
    print(f"Base solweig dir: {args.solweig_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
