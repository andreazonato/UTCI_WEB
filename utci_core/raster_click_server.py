#!/usr/bin/env python3
import argparse
import json
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import rasterio
from rasterio.warp import transform

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


def sample_timeseries(ds, lon: float, lat: float) -> list:
    if ds.crs is None:
        raise ValueError("Dataset CRS is missing.")
    if ds.crs.to_epsg() != 4326:
        xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
        x, y = xs[0], ys[0]
    else:
        x, y = lon, lat
    values = next(ds.sample([(x, y)], indexes=list(range(1, ds.count + 1))))
    out = []
    nodata = ds.nodata
    for v in values:
        if v is None:
            out.append(None)
            continue
        try:
            vf = float(v)
        except Exception:
            out.append(None)
            continue
        if nodata is not None and vf == nodata:
            out.append(None)
        elif not (vf == vf):  # NaN
            out.append(None)
        elif vf <= NODATA_FLOOR:
            out.append(None)
        else:
            out.append(vf)
    return out


def sample_value(ds, lon: float, lat: float) -> float | None:
    if ds.crs is None:
        raise ValueError("Dataset CRS is missing.")
    if ds.crs.to_epsg() != 4326:
        xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
        x, y = xs[0], ys[0]
    else:
        x, y = lon, lat
    v = next(ds.sample([(x, y)]))[0]
    nodata = ds.nodata
    try:
        vf = float(v)
    except Exception:
        return None
    if nodata is not None and vf == nodata:
        return None
    if not (vf == vf):
        return None
    if vf <= NODATA_FLOOR:
        return None
    return vf


def sample_mean_value(ds, lon: float, lat: float) -> float | None:
    if ds.count <= 1:
        return sample_value(ds, lon, lat)
    if ds.crs is None:
        raise ValueError("Dataset CRS is missing.")
    if ds.crs.to_epsg() != 4326:
        xs, ys = transform("EPSG:4326", ds.crs, [lon], [lat])
        x, y = xs[0], ys[0]
    else:
        x, y = lon, lat
    values = next(ds.sample([(x, y)], indexes=list(range(1, ds.count + 1))))
    nodata = ds.nodata
    acc = 0.0
    cnt = 0
    for v in values:
        try:
            vf = float(v)
        except Exception:
            continue
        if nodata is not None and vf == nodata:
            continue
        if not (vf == vf):
            continue
        if vf <= NODATA_FLOOR:
            continue
        try:
            acc += vf
            cnt += 1
        except Exception:
            continue
    if cnt == 0:
        return None
    return acc / cnt


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
                values = sample_timeseries(ds, lng_f, lat_f)
                payload = {"values": values}
            else:
                if mean == "1":
                    value = sample_mean_value(ds, lng_f, lat_f)
                else:
                    value = sample_value(ds, lng_f, lat_f)
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
