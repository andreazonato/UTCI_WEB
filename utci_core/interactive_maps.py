#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
import matplotlib as mpl
from PIL import Image
import folium
from folium import Element
from folium.plugins import GroupedLayerControl


DATE_YYYYMMDD_RE = re.compile(r"^(?P<var>[A-Za-z0-9]+)_(?P<date>[0-9]{8})\.tif$", re.IGNORECASE)
TIME_META_RE = re.compile(r"(?P<y>[0-9]{4})[-/](?P<m>[0-9]{2})[-/](?P<d>[0-9]{2})[ T]+(?P<h>[0-9]{2}):(?P<mi>[0-9]{2})(:(?P<s>[0-9]{2}))?")

LEAFLET_VERSION = "1.9.3"
GROUPED_LAYER_CONTROL_VERSION = "0.6.1"
DEFAULT_RENDER_MAX_DIM = 900
DEFAULT_STATS_MAX_DIM = 300
# Treat extreme negatives (e.g. -9999 nodata, plus resampling bleed) as NaN.
NODATA_FLOOR = -1000.0

DEFAULT_HOURLY_VARS: dict[str, dict] = {
    "UTCI": {"cmap": "RdBu_r", "resampling": Resampling.bilinear, "opacity": 0.7},
    "Ta": {"cmap": "RdBu_r", "resampling": Resampling.bilinear, "opacity": 0.7},
    "TMRT": {"cmap": "RdBu_r", "resampling": Resampling.bilinear, "opacity": 0.7},
    "Va10m": {"cmap": "RdBu_r", "resampling": Resampling.bilinear, "opacity": 0.7},
}

SINGLE_BAND_LAYERS = [
    ("SVF_static.tif", "SVF static", "RdBu_r", Resampling.bilinear, 0.7),
]

UTCI_CATEGORIES = [
    {"range": (-1e9, -40.0), "label": "Extreme cold stress", "color": "#08306b", "range_label": "< -40"},
    {"range": (-40.0, -27.0), "label": "Very strong cold stress", "color": "#08519c", "range_label": "-40 to -27"},
    {"range": (-27.0, -13.0), "label": "Strong cold stress", "color": "#2171b5", "range_label": "-27 to -13"},
    {"range": (-13.0, 0.0), "label": "Moderate cold stress", "color": "#4292c6", "range_label": "-13 to 0"},
    {"range": (0.0, 9.0), "label": "Slight cold stress", "color": "#6baed6", "range_label": "0 to 9"},
    {"range": (9.0, 26.0), "label": "No thermal stress", "color": "#239223", "range_label": "9 to 26"},
    {"range": (26.0, 32.0), "label": "Moderate heat stress", "color": "#FFA500", "range_label": "26 to 32"},
    {"range": (32.0, 38.0), "label": "Strong heat stress", "color": "#FF4500", "range_label": "32 to 38"},
    {"range": (38.0, 46.0), "label": "Very strong heat stress", "color": "#FF0000", "range_label": "38 to 46"},
    {"range": (46.0, 1e9), "label": "Extreme heat stress", "color": "#800000", "range_label": "> 46"},
]

def _log(msg: str, *, verbose: bool) -> None:
    if not verbose:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [interactive_maps] {msg}", flush=True)


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def parse_hourly_vars(spec: str | None) -> dict[str, dict]:
    """Select hourly vars preserving DEFAULT_HOURLY_VARS ordering."""
    if spec is None or not str(spec).strip():
        return dict(DEFAULT_HOURLY_VARS)
    req = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not req:
        return dict(DEFAULT_HOURLY_VARS)
    known = {k.lower(): k for k in DEFAULT_HOURLY_VARS.keys()}
    selected: list[str] = []
    unknown: list[str] = []
    for item in req:
        key = known.get(item.lower())
        if key is None:
            unknown.append(item)
        elif key not in selected:
            selected.append(key)
    if unknown:
        raise ValueError(
            f"Unknown hourly vars: {', '.join(unknown)}. "
            f"Available: {', '.join(DEFAULT_HOURLY_VARS.keys())}"
        )
    return {k: DEFAULT_HOURLY_VARS[k] for k in DEFAULT_HOURLY_VARS.keys() if k in selected}


def discover_dates_compact(folder: Path, vars_to_check: list[str] | None = None) -> list[str]:
    vars_to_check = vars_to_check or list(DEFAULT_HOURLY_VARS.keys())
    dates: set[str] = set()
    for var in vars_to_check:
        for tif in folder.glob(f"{var}_*.tif"):
            m = DATE_YYYYMMDD_RE.match(tif.name)
            if not m:
                continue
            if m.group("var").upper() != var.upper():
                continue
            dates.add(m.group("date"))
    return sorted(dates)


def date_label(date_compact: str) -> str:
    if len(date_compact) == 8 and date_compact.isdigit():
        return f"{date_compact[0:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
    return date_compact


def get_bounds_latlon_ds(ds: rasterio.io.DatasetReader) -> list[list[float]]:
    minx, miny, maxx, maxy = ds.bounds
    crs = ds.crs
    if crs is None:
        raise ValueError("Raster is missing CRS.")
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg != 4326:
        minx, miny, maxx, maxy = transform_bounds(crs, "EPSG:4326", minx, miny, maxx, maxy)
    return [[miny, minx], [maxy, maxx]]


def _downsample_shape(height: int, width: int, max_dim: int) -> tuple[int, int]:
    if max_dim <= 0:
        return height, width
    scale = max(height, width) / max_dim
    if scale <= 1:
        return height, width
    return max(1, int(round(height / scale))), max(1, int(round(width / scale)))

def _parse_time_meta(raw: str) -> tuple[str, int, int] | None:
    if not raw:
        return None
    m = TIME_META_RE.search(str(raw))
    if not m:
        return None
    y = m.group("y")
    mo = m.group("m")
    d = m.group("d")
    h = int(m.group("h"))
    mi = int(m.group("mi"))
    date_compact = f"{y}{mo}{d}"
    return date_compact, h, mi

def _band_time_info(ds: rasterio.io.DatasetReader, fallback_date: str) -> list[dict]:
    infos: list[dict] = []
    descs = list(ds.descriptions or [])
    for i in range(1, int(ds.count or 1) + 1):
        raw = ""
        try:
            tags = ds.tags(i) or {}
            raw = tags.get("TIME") or ""
        except Exception:
            raw = ""
        if not raw and descs and len(descs) >= i:
            raw = descs[i - 1] or ""
        parsed = _parse_time_meta(raw) if raw else None
        if parsed:
            dc, hh, mm = parsed
            label = f"{date_label(dc)} {hh:02d}:{mm:02d}"
            infos.append(
                {
                    "band_index0": i - 1,
                    "date_compact": dc,
                    "hour": hh,
                    "minute": mm,
                    "label": label,
                }
            )
        else:
            infos.append(
                {
                    "band_index0": i - 1,
                    "date_compact": fallback_date,
                    "hour": i - 1,
                    "minute": 0,
                    "label": f"{date_label(fallback_date)} {i - 1:02d}:00",
                }
            )
    return infos

def _date_band_timeline(solweig_dir: Path, dc: str, var_names: list[str]) -> list[dict]:
    for var_name in var_names:
        tif_path = solweig_dir / f"{var_name}_{dc}.tif"
        if not tif_path.exists():
            continue
        try:
            with rasterio.open(tif_path) as ds:
                return _band_time_info(ds, dc)
        except Exception:
            continue
    return []

def _apply_nodata(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32, copy=False)
    if nodata is not None:
        if np.isnan(nodata):
            arr = np.where(np.isnan(arr), np.nan, arr)
        else:
            arr = np.where(arr == nodata, np.nan, arr)
    # Some rasters omit nodata metadata or smear -9999 via resampling.
    arr = np.where(arr <= NODATA_FLOOR, np.nan, arr)
    return arr


def compute_min_max_from_dataset(
    ds: rasterio.io.DatasetReader,
    bands: list[int] | None = None,
    max_dim: int = DEFAULT_STATS_MAX_DIM,
    resampling: Resampling = Resampling.nearest,
) -> tuple[float, float] | None:
    try:
        out_h, out_w = _downsample_shape(ds.height, ds.width, max_dim)
        nodata = ds.nodata
        band_list = bands if bands else list(range(1, ds.count + 1))
        vmin = None
        vmax = None
        for b in band_list:
            arr = ds.read(b, out_shape=(out_h, out_w), resampling=resampling)
            arr = _apply_nodata(arr, nodata)
            if np.all(np.isnan(arr)):
                continue
            a_min = float(np.nanmin(arr))
            a_max = float(np.nanmax(arr))
            vmin = a_min if vmin is None else min(vmin, a_min)
            vmax = a_max if vmax is None else max(vmax, a_max)
        if vmin is None or vmax is None:
            return None
        if vmin == vmax:
            vmax = vmin + 1e-6
        return vmin, vmax
    except Exception:
        return None


def compute_min_max_from_raster(tif_path: Path, bands: list[int] | None = None, max_dim: int = DEFAULT_STATS_MAX_DIM) -> tuple[float, float] | None:
    try:
        with rasterio.open(tif_path) as ds:
            return compute_min_max_from_dataset(ds, bands=bands, max_dim=max_dim)
    except Exception:
        return None


def compute_mean_min_max_from_raster(tif_path: Path, bands: list[int] | None = None, max_dim: int = 300) -> tuple[float, float] | None:
    try:
        with rasterio.open(tif_path) as ds:
            out_h, out_w = _downsample_shape(ds.height, ds.width, max_dim)
            nodata = ds.nodata
            band_list = bands if bands else list(range(1, ds.count + 1))
            sum_arr = None
            count_arr = None
        for b in band_list:
            arr = ds.read(b, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
            arr = _apply_nodata(arr, nodata)
            if sum_arr is None:
                sum_arr = np.zeros_like(arr, dtype=np.float64)
                count_arr = np.zeros_like(arr, dtype=np.int32)
                valid = np.isfinite(arr)
                sum_arr[valid] += arr[valid]
                count_arr[valid] += 1
            if sum_arr is None or count_arr is None:
                return None
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_arr = sum_arr / np.where(count_arr == 0, np.nan, count_arr)
            if np.all(np.isnan(mean_arr)):
                return None
            vmin = float(np.nanmin(mean_arr))
            vmax = float(np.nanmax(mean_arr))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return None
            if vmin == vmax:
                vmax = vmin + 1e-6
            return vmin, vmax
    except Exception:
        return None


@lru_cache(maxsize=32)
def colormap_to_hex_list(cmap_name: str, n: int = 256) -> list[str]:
    try:
        cmap = mpl.colormaps.get_cmap(cmap_name)
        if hasattr(cmap, "resampled"):
            cmap = cmap.resampled(n)
    except Exception:
        cmap = mpl.cm.get_cmap(cmap_name, n)
    colors = []
    for i in range(n):
        r, g, b, _ = cmap(i)
        colors.append(mpl.colors.rgb2hex((r, g, b)))
    return colors


@lru_cache(maxsize=32)
def _get_cmap(cmap_name: str):
    try:
        return mpl.colormaps.get_cmap(cmap_name)
    except Exception:
        return mpl.cm.get_cmap(cmap_name)


def unit_for_var(var_name: str) -> str:
    v = var_name.lower()
    if v in ("utci", "ta", "tmrt"):
        return "°C"
    if v in ("va10m",):
        return "m/s"
    return ""


def save_png_from_array(arr: np.ndarray, cmap_name: str, vmin: float, vmax: float, out_png: Path, overwrite: bool) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.exists() and not overwrite:
        return
    arr = _apply_nodata(arr, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (arr - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    t = np.where(np.isfinite(t), t, np.nan)

    cmap = _get_cmap(cmap_name)
    rgba = cmap(np.where(np.isfinite(t), t, 0.0), bytes=True)  # (H,W,4) uint8
    rgba[..., 3] = np.where(np.isfinite(t), 255, 0).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(out_png)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hc = hex_color.strip().lstrip("#")
    if len(hc) != 6:
        return (0, 0, 0)
    return (int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16))


def categorize_utci_rgba(arr: np.ndarray) -> np.ndarray:
    """
    Map UTCI values to categorical RGBA colors.
    Values below the first threshold are mapped to the first category (No Heat Stress).
    """
    a = _apply_nodata(arr, None)
    out = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.uint8)
    valid = np.isfinite(a)
    out[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    if not np.any(valid):
        return out

    cats = UTCI_CATEGORIES
    # Initialize all valid pixels to first category
    r0, g0, b0 = _hex_to_rgb(cats[0]["color"])
    out[valid, 0] = r0
    out[valid, 1] = g0
    out[valid, 2] = b0

    for idx, cat in enumerate(cats):
        lo, hi = cat["range"]
        r, g, b = _hex_to_rgb(cat["color"])
        if idx == len(cats) - 1:
            m = valid & (a >= lo)
        else:
            m = valid & (a >= lo) & (a < hi)
        out[m, 0] = r
        out[m, 1] = g
        out[m, 2] = b
    return out


def save_png_from_rgba(rgba: np.ndarray, out_png: Path, overwrite: bool) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.exists() and not overwrite:
        return
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(out_png)

def transparent_png_data_url() -> str:
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def _download(url: str, dest: Path, timeout_s: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "interactive_maps.py"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    dest.write_bytes(data)


def ensure_offline_assets(assets_dir: Path) -> bool:
    """
    Download the minimal JS/CSS (Leaflet + GroupedLayerControl) into assets_dir so the
    generated HTML can be opened without internet access.
    """
    leaflet_base = f"https://cdn.jsdelivr.net/npm/leaflet@{LEAFLET_VERSION}/dist"
    glc_base = (
        "https://cdnjs.cloudflare.com/ajax/libs/leaflet-groupedlayercontrol/"
        f"{GROUPED_LAYER_CONTROL_VERSION}"
    )
    items = [
        (f"{leaflet_base}/leaflet.js", assets_dir / "leaflet.js"),
        (f"{leaflet_base}/leaflet.css", assets_dir / "leaflet.css"),
        (f"{glc_base}/leaflet.groupedlayercontrol.min.js", assets_dir / "leaflet.groupedlayercontrol.min.js"),
        (f"{glc_base}/leaflet.groupedlayercontrol.min.css", assets_dir / "leaflet.groupedlayercontrol.min.css"),
        # Leaflet CSS references these relative images.
        (f"{leaflet_base}/images/layers.png", assets_dir / "images" / "layers.png"),
        (f"{leaflet_base}/images/layers-2x.png", assets_dir / "images" / "layers-2x.png"),
        (f"{leaflet_base}/images/marker-icon.png", assets_dir / "images" / "marker-icon.png"),
        (f"{leaflet_base}/images/marker-icon-2x.png", assets_dir / "images" / "marker-icon-2x.png"),
        (f"{leaflet_base}/images/marker-shadow.png", assets_dir / "images" / "marker-shadow.png"),
    ]

    ok = True
    for url, dest in items:
        try:
            _download(url, dest)
        except Exception as e:
            ok = False
            print(f"[warn] Cannot download offline asset {url}: {e}")
    return ok


def configure_folium_offline_assets(*, html_dir: Path, assets_dir: Path) -> callable:
    """
    Point folium's JS/CSS links to local assets. Returns a restore() callback.
    """
    import folium.folium as folium_map_mod

    orig_map_js = list(folium_map_mod.Map.default_js)
    orig_map_css = list(folium_map_mod.Map.default_css)
    orig_glc_js = list(GroupedLayerControl.default_js)
    orig_glc_css = list(GroupedLayerControl.default_css)

    rel_base = os.path.relpath(str(assets_dir), start=str(html_dir)).replace(os.sep, "/")
    folium_map_mod.Map.default_js = [("leaflet", f"{rel_base}/leaflet.js")]
    folium_map_mod.Map.default_css = [("leaflet_css", f"{rel_base}/leaflet.css")]
    GroupedLayerControl.default_js = [("leaflet.groupedlayercontrol.min.js", f"{rel_base}/leaflet.groupedlayercontrol.min.js")]
    GroupedLayerControl.default_css = [("leaflet.groupedlayercontrol.min.css", f"{rel_base}/leaflet.groupedlayercontrol.min.css")]

    def restore() -> None:
        folium_map_mod.Map.default_js = orig_map_js
        folium_map_mod.Map.default_css = orig_map_css
        GroupedLayerControl.default_js = orig_glc_js
        GroupedLayerControl.default_css = orig_glc_css

    return restore


def build_raster_map(
    solweig_dir: Path,
    output_dir: Path,
    html_dir: Path,
    default_hour: int,
    overwrite_png: bool,
    raster_click_mode: str,
    raster_click_server_url: str,
    raster_click_folder: str,
    *,
    verbose: bool = True,
    render_max_dim: int = DEFAULT_RENDER_MAX_DIM,
    stats_max_dim: int = DEFAULT_STATS_MAX_DIM,
    hourly_vars: dict[str, dict] | None = None,
) -> folium.Map:
    hourly_vars = dict(hourly_vars or DEFAULT_HOURLY_VARS)
    var_names = list(hourly_vars.keys())
    if not var_names:
        raise ValueError("No hourly vars selected.")
    t0 = time.perf_counter()
    dates = discover_dates_compact(solweig_dir, vars_to_check=var_names)
    if not dates:
        raise FileNotFoundError(
            f"No hourly rasters found in {solweig_dir} for vars: {', '.join(var_names)}."
        )
    _log(f"Discovered {len(dates)} day file(s): {', '.join(dates)}", verbose=verbose)

    # Build a unified timeline across all dates using band metadata when available.
    date_band_meta: dict[str, list[dict]] = {}
    for dc in dates:
        meta_list = _date_band_timeline(solweig_dir, dc, var_names)
        if not meta_list:
            # Fallback: use band count from first available var.
            max_hours = 0
            for var_name in var_names:
                tif_path = solweig_dir / f"{var_name}_{dc}.tif"
                if not tif_path.exists():
                    continue
                try:
                    with rasterio.open(tif_path) as ds:
                        max_hours = max(max_hours, min(24, int(ds.count or 1)))
                except Exception:
                    continue
            if max_hours > 0:
                meta_list = [
                    {
                        "band_index0": h,
                        "date_compact": dc,
                        "hour": h,
                        "minute": 0,
                        "label": f"{date_label(dc)} {h:02d}:00",
                    }
                    for h in range(max_hours)
                ]
        if meta_list:
            date_band_meta[dc] = meta_list[:24]

    timeline: list[tuple[str, int]] = []
    timeline_labels: list[str] = []
    timeline_dates: list[str] = []
    timeline_hours: list[int] = []
    for dc in dates:
        meta_list = date_band_meta.get(dc, [])
        for meta in meta_list:
            h = int(meta.get("band_index0", 0))
            timeline.append((dc, h))
            timeline_labels.append(str(meta.get("label", f"{date_label(dc)} {h:02d}:00")))
            timeline_dates.append(str(meta.get("date_compact", dc)))
            timeline_hours.append(int(meta.get("hour", h)))
    if not timeline:
        raise FileNotFoundError(f"No hourly bands found in {solweig_dir}.")
    try:
        default_index = next(i for i, h in enumerate(timeline_hours) if h == default_hour)
    except Exception:
        default_index = 0

    m = folium.Map(location=[0, 0], zoom_start=2, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("cartodbpositron", name="Positron").add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="Satellite", attr="Esri").add_to(m)

    grouped_layers: dict[str, list] = {"Hourly": [], "UTCI categories": []}
    hourly_layers_js: list[dict] = []
    map_bounds = None
    folder_name = (raster_click_folder or ".").strip() or "."
    show_first_overlay = True

    png_map: dict[tuple[str, str, int], Path] = {}
    cat_map: dict[tuple[str, int], Path] = {}
    var_stats: dict[str, tuple[float, float]] = {}

    # Hourly + mean (per day)
    for dc in dates:
        dl = date_label(dc)
        _log(f"Processing date {dc} ({dl})", verbose=verbose)
        for var_name, meta in hourly_vars.items():
            tif_path = solweig_dir / f"{var_name}_{dc}.tif"
            if not tif_path.exists():
                continue

            _log(f"{dc} {var_name}: opening {tif_path.name}", verbose=verbose)
            with rasterio.open(tif_path) as ds:
                bounds = get_bounds_latlon_ds(ds)
                if map_bounds is None:
                    map_bounds = bounds

                n_bands = int(ds.count or 1)
                n_hours = min(24, n_bands)
                band_list = list(range(1, n_hours + 1))

                out_h, out_w = _downsample_shape(ds.height, ds.width, render_max_dim)
                cat_pngs: list[Path] | None = None
                with WarpedVRT(
                    ds,
                    crs="EPSG:4326",
                    resampling=meta["resampling"],
                    width=out_w,
                    height=out_h,
                ) as vrt:
                    stats = compute_min_max_from_dataset(vrt, bands=band_list, max_dim=stats_max_dim)
                    if stats is None:
                        _log(f"{dc} {var_name}: cannot compute min/max (fallback to 0..1)", verbose=verbose)
                        stats = (0.0, 1.0)
                    vmin, vmax = stats
                    if var_name in var_stats:
                        cur_min, cur_max = var_stats[var_name]
                        var_stats[var_name] = (min(cur_min, vmin), max(cur_max, vmax))
                    else:
                        var_stats[var_name] = (vmin, vmax)
                    _log(
                        f"{dc} {var_name}: bands={n_bands} hours={n_hours} vmin={vmin:.3f} vmax={vmax:.3f} "
                        f"render_dim={out_w}x{out_h}",
                        verbose=verbose,
                    )
                    out_dir = output_dir / "raster_images" / dc / var_name
                    hour_pngs = [out_dir / f"hour_{h:02d}.png" for h in range(n_hours)]
                    if overwrite_png:
                        to_generate = list(range(n_hours))
                    else:
                        to_generate = [h for h in range(n_hours) if not hour_pngs[h].exists()]
                    if to_generate:
                        _log(f"{dc} {var_name}: generating {len(to_generate)} PNG(s) in {out_dir}", verbose=verbose)
                        for h in to_generate:
                            arr = vrt.read(h + 1)
                            save_png_from_array(arr, meta["cmap"], vmin, vmax, hour_pngs[h], overwrite=overwrite_png)
                    else:
                        _log(f"{dc} {var_name}: PNG(s) already present (skip)", verbose=verbose)
                    for h in range(n_hours):
                        png_map[(var_name, dc, h)] = hour_pngs[h]

                    # UTCI categories (hourly)
                    if var_name == "UTCI":
                        cat_out_dir = output_dir / "raster_images" / dc / "UTCI_categories"
                        cat_pngs = [cat_out_dir / f"hour_{h:02d}.png" for h in range(n_hours)]
                        if overwrite_png:
                            cat_generate = list(range(n_hours))
                        else:
                            cat_generate = [h for h in range(n_hours) if not cat_pngs[h].exists()]
                        if cat_generate:
                            _log(
                                f"{dc} UTCI_categories: generating {len(cat_generate)} PNG(s) in {cat_out_dir}",
                                verbose=verbose,
                            )
                            for h in cat_generate:
                                rgba = categorize_utci_rgba(vrt.read(h + 1))
                                save_png_from_rgba(rgba, cat_pngs[h], overwrite=overwrite_png)
                        else:
                            _log(f"{dc} UTCI_categories: PNG(s) already present (skip)", verbose=verbose)
                        for h in range(n_hours):
                            cat_map[(dc, h)] = cat_pngs[h]

    transparent_png = output_dir / "raster_images" / "transparent.png"
    if not transparent_png.exists():
        transparent_png.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(transparent_png)

    def _rel_url(p: Path) -> str:
        try:
            rel = os.path.relpath(str(p), start=str(html_dir)).replace(os.sep, "/")
            return rel
        except Exception:
            return p.resolve().as_uri()

    transparent_url = _rel_url(transparent_png)

    def _png_exists(p: Path | None) -> bool:
        try:
            return bool(p and p.exists())
        except Exception:
            return False

    def _folium_image_path(p: Path | None) -> str:
        if _png_exists(p):
            return str(p.resolve())
        return str(transparent_png.resolve())

    for var_name, meta in hourly_vars.items():
        if not any(k[0] == var_name for k in png_map.keys()):
            continue
        urls: list[str] = []
        for dc, h in timeline:
            p = png_map.get((var_name, dc, h))
            urls.append(_rel_url(p) if _png_exists(p) else transparent_url)
        vmin, vmax = var_stats.get(var_name, (0.0, 1.0))
        layer_name = f"{var_name} hourly"
        init_idx = min(default_index, len(timeline) - 1)
        init_dc, init_h = timeline[init_idx]
        init_png = png_map.get((var_name, init_dc, init_h))
        overlay = folium.raster_layers.ImageOverlay(
            image=_folium_image_path(init_png),
            bounds=map_bounds,
            opacity=meta["opacity"],
            name=layer_name,
            interactive=False,
            cross_origin=False,
            zindex=2,
            show=show_first_overlay,
        )
        overlay.add_to(m)
        grouped_layers["Hourly"].append(overlay)
        show_first_overlay = False
        hourly_layers_js.append(
            {
                "layer_var": overlay.get_name(),
                "urls": urls,
                "opacity": meta["opacity"],
                "colors": colormap_to_hex_list(meta["cmap"]),
                "vmin": vmin,
                "vmax": vmax,
                "name": layer_name,
                "unit": unit_for_var(var_name),
                "scenario_folder": folder_name,
                "var_name": var_name,
            }
        )

        # UTCI categories (hourly)
        if var_name == "UTCI" and cat_map:
            cat_urls: list[str] = []
            for dc, h in timeline:
                p = cat_map.get((dc, h))
                cat_urls.append(_rel_url(p) if _png_exists(p) else transparent_url)
            cat_layer_name = "UTCI categories hourly"
            init_idx = min(default_index, len(timeline) - 1)
            init_dc, init_h = timeline[init_idx]
            init_cat = cat_map.get((init_dc, init_h))
            cat_overlay = folium.raster_layers.ImageOverlay(
                image=_folium_image_path(init_cat),
                bounds=map_bounds,
                opacity=meta["opacity"],
                name=cat_layer_name,
                interactive=False,
                cross_origin=False,
                zindex=2,
                show=False,
            )
            cat_overlay.add_to(m)
            grouped_layers["UTCI categories"].append(cat_overlay)
            hourly_layers_js.append(
                {
                    "layer_var": cat_overlay.get_name(),
                    "urls": cat_urls,
                    "opacity": meta["opacity"],
                    "colors": [c["color"] for c in UTCI_CATEGORIES],
                    "vmin": None,
                    "vmax": None,
                    "categories": UTCI_CATEGORIES,
                    "name": cat_layer_name,
                    "unit": "°C",
                    "scenario_folder": folder_name,
                    "var_name": "UTCI",
                    "is_categories": True,
                }
            )

    if map_bounds:
        m.fit_bounds(map_bounds)
        _log("Map bounds set from first raster.", verbose=verbose)

    if grouped_layers:
        GroupedLayerControl(groups=grouped_layers, collapsed=False).add_to(m)
    else:
        folium.LayerControl(collapsed=False).add_to(m)
    folium.LayerControl(position="bottomright", collapsed=False).add_to(m)

    add_controls_and_popups(
        m,
        hourly_layers_js,
        default_index,
        timeline_labels,
        timeline_dates,
        timeline_hours,
        raster_click_mode=raster_click_mode,
        raster_click_server_url=raster_click_server_url,
    )
    _log(f"Prepared {len(hourly_layers_js)} overlay layer(s) in {time.perf_counter() - t0:.2f}s.", verbose=verbose)
    return m


def add_controls_and_popups(
    m: folium.Map,
    hourly_layers_js: list[dict],
    default_index: int,
    timeline_labels: list[str],
    timeline_dates: list[str],
    timeline_hours: list[int],
    raster_click_mode: str,
    raster_click_server_url: str,
) -> None:
    if not hourly_layers_js:
        return

    slider_html = f"""
    <style>
        .fancy-panel {{
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 10px;
            padding: 10px 12px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
            font-size: 13px;
            font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            min-width: 380px;
        }}
        .fancy-panel .panel-title {{
            font-weight: 700;
            font-size: 13px;
            margin-bottom: 6px;
            letter-spacing: 0.2px;
        }}
        .fancy-panel .panel-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 6px;
        }}
        .fancy-panel .panel-row:last-child {{
            margin-bottom: 0;
        }}
        .fancy-panel .muted {{
            color: #666;
            font-size: 12px;
        }}
        .fancy-panel input[type=range] {{
            accent-color: #2c7fb8;
        }}
        .fancy-panel button {{
            border: 1px solid #ccc;
            background: #f7f7f7;
            border-radius: 6px;
            padding: 4px 8px;
            cursor: pointer;
        }}
        .fancy-panel button:hover {{
            background: #efefef;
        }}
        .fancy-panel .colorbar-item {{
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 6px 8px;
            margin-bottom: 8px;
            background: #fafafa;
            cursor: pointer;
        }}
        .fancy-panel .colorbar-item:hover {{
            border-color: #bbb;
            background: #f5f5f5;
        }}
    </style>
    <div id="hour-control" class="fancy-panel fancy-layer-panel">
        <div class="panel-title">Layer controls</div>
        <div class="panel-row">
            <label for="hour-slider">Time:</label>
            <input type="range" min="0" max="{max(0, len(timeline_labels) - 1)}" value="{default_index}" step="1" id="hour-slider">
            <span id="hour-value">{timeline_labels[default_index] if timeline_labels else ''}</span>
        </div>
        <div class="panel-row">
            <label for="opacity-slider">Opacity:</label>
            <input type="range" min="0" max="1" value="0.7" step="0.05" id="opacity-slider">
            <span id="opacity-value">0.70</span>
            <button id="raster-hide-all">Hide all</button>
        </div>
        <div class="panel-row muted">Click a colorbar to hide the layer.</div>
        <div id="raster-colorbars"></div>
    </div>
    """
    m.get_root().html.add_child(Element(slider_html))

    layer_entries = []
    for entry in hourly_layers_js:
        layer_entries.append(
            "{layer: "
            + entry["layer_var"]
            + ", urls: "
            + json.dumps(entry["urls"])
            + ", colors: "
            + json.dumps(entry.get("colors", []))
            + ", vmin: "
            + json.dumps(entry.get("vmin", None))
            + ", vmax: "
            + json.dumps(entry.get("vmax", None))
            + ", categories: "
            + json.dumps(entry.get("categories", None))
            + ", is_categories: "
            + json.dumps(bool(entry.get("is_categories", False)))
            + ", name: "
            + json.dumps(entry.get("name", "Layer"))
            + ", unit: "
            + json.dumps(entry.get("unit", ""))
            + ", scenario_folder: "
            + json.dumps(entry.get("scenario_folder", "."))
            + ", var_name: "
            + json.dumps(entry.get("var_name", ""))
            + "}"
        )
    layers_js = "[\n" + ",\n".join(layer_entries) + "\n]"
    # NOTE: folium wraps everything in `m.get_root().script` inside a single <script> tag.
    # Do not include nested <script> tags here, otherwise the generated HTML breaks with:
    # "Uncaught SyntaxError: Unexpected token '<'".
    js = f"""
    window.addEventListener('load', function() {{
        var map = {m.get_name()};
        var hourlyRasterLayers = {layers_js};
        var timelineLabels = {json.dumps(timeline_labels)};
        var timelineDates = {json.dumps(timeline_dates)};
        var timelineHours = {json.dumps(timeline_hours)};
        var currentIndex = {default_index};
        var currentHour = timelineHours[currentIndex] || 0;
        var currentDateCompact = timelineDates[currentIndex] || '';
        var currentOpacity = 0.7;
        var rasterClickMode = {json.dumps(raster_click_mode)};
        var rasterClickServerUrl = {json.dumps(raster_click_server_url)};
        var utciBands = {json.dumps(UTCI_CATEGORIES)};
        var timelineDatesUnique = [];
        for (var i = 0; i < timelineDates.length; i++) {{
            var d = timelineDates[i];
            if (!d) continue;
            if (timelineDatesUnique.indexOf(d) === -1) timelineDatesUnique.push(d);
        }}
        var dayHoursMap = {{}};
        for (var i = 0; i < timelineDates.length; i++) {{
            var d = timelineDates[i];
            if (!d) continue;
            if (!dayHoursMap[d]) dayHoursMap[d] = [];
            dayHoursMap[d].push(timelineHours[i]);
        }}

        function pad2(n) {{ return String(n).padStart(2, '0'); }}
        function labelDateCompact(dc) {{
            if (!dc || dc.length !== 8) return dc;
            return dc.slice(0, 4) + '-' + dc.slice(4, 6) + '-' + dc.slice(6, 8);
        }}

        function renderColorbar(el, vmin, vmax, colors, name, unit) {{
            var title = document.createElement('div');
            title.textContent = (name || 'Layer') + (unit ? (' [' + unit + ']') : '');
            title.style.fontWeight = '600';
            title.style.marginBottom = '2px';
            el.appendChild(title);

            var canvas = document.createElement('canvas');
            canvas.width = 420;
            canvas.height = 12;
            canvas.style.border = '1px solid #ccc';
            el.appendChild(canvas);

            var ctx = canvas.getContext('2d');
            var grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
            for (var i = 0; i < colors.length; i++) {{
                grad.addColorStop(i / (colors.length - 1), colors[i]);
            }}
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            var labels = document.createElement('div');
            labels.style.display = 'flex';
            labels.style.justifyContent = 'space-between';
            var hasRange = !(vmin === null || vmax === null || !isFinite(vmin) || !isFinite(vmax));
            var minEl = document.createElement('span');
            minEl.textContent = hasRange ? (Number(vmin).toFixed(2) + (unit ? (' ' + unit) : '')) : 'n/a';
            var maxEl = document.createElement('span');
            maxEl.textContent = hasRange ? (Number(vmax).toFixed(2) + (unit ? (' ' + unit) : '')) : 'n/a';
            labels.appendChild(minEl);
            labels.appendChild(maxEl);
            el.appendChild(labels);
        }}

        function renderCategoryLegend(el, categories) {{
            if (!categories || !categories.length) return;
            categories.forEach(function(c) {{
                var row = document.createElement('div');
                row.style.display = 'flex';
                row.style.alignItems = 'center';
                row.style.gap = '8px';
                row.style.marginTop = '3px';
                var sw = document.createElement('span');
                sw.style.display = 'inline-block';
                sw.style.width = '14px';
                sw.style.height = '10px';
                sw.style.border = '1px solid #999';
                sw.style.background = c.color || '#ccc';
                var txt = document.createElement('span');
                txt.style.fontSize = '11px';
                var rng = c.range_label || '';
                if (!rng && c.range && c.range.length === 2) {{
                    rng = c.range[0] + '–' + c.range[1];
                }}
                txt.textContent = (c.label || 'Category') + (rng ? (' (' + rng + ')') : '');
                row.appendChild(sw);
                row.appendChild(txt);
                el.appendChild(row);
            }});
        }}

        function renderRasterColorbars() {{
            var container = document.getElementById('raster-colorbars');
            if (!container) return;
            container.innerHTML = '';
            var active = [];
            hourlyRasterLayers.forEach(function(item) {{
                if (map.hasLayer(item.layer)) active.push(item);
            }});
            if (!active.length) {{
                var empty = document.createElement('div');
                empty.className = 'muted';
                empty.textContent = 'No active layer.';
                container.appendChild(empty);
                return;
            }}
            active.forEach(function(item) {{
                var wrap = document.createElement('div');
                wrap.className = 'colorbar-item';
                wrap.title = 'Click to hide the layer';
                wrap.addEventListener('click', function(ev) {{
                    ev.stopPropagation();
                    if (map.hasLayer(item.layer)) map.removeLayer(item.layer);
                    renderRasterColorbars();
                }});
                if (item.categories && item.categories.length) {{
                    var title = document.createElement('div');
                    title.textContent = (item.name || 'Layer') + (item.unit ? (' [' + item.unit + ']') : '');
                    title.style.fontWeight = '600';
                    title.style.marginBottom = '2px';
                    wrap.appendChild(title);
                    renderCategoryLegend(wrap, item.categories);
                }} else {{
                    renderColorbar(wrap, item.vmin, item.vmax, item.colors || [], item.name, item.unit);
                }}
                container.appendChild(wrap);
            }});
        }}

        function updateRasterHour(h) {{
            currentIndex = h;
            currentHour = timelineHours[currentIndex] || 0;
            currentDateCompact = timelineDates[currentIndex] || '';
            var label = document.getElementById('hour-value');
            if (label) label.textContent = timelineLabels[currentIndex] || (pad2(currentHour) + ':00');
            hourlyRasterLayers.forEach(function(item) {{
                if (item.urls && item.urls.length > currentIndex) item.layer.setUrl(item.urls[currentIndex]);
            }});
            renderRasterColorbars();
        }}

        function updateRasterOpacity(val) {{
            currentOpacity = val;
            var label = document.getElementById('opacity-value');
            if (label) label.textContent = Number(val).toFixed(2);
            hourlyRasterLayers.forEach(function(item) {{
                if (item.layer && item.layer.setOpacity) item.layer.setOpacity(val);
            }});
            renderRasterColorbars();
        }}

        function hideAllOverlays() {{
            hourlyRasterLayers.forEach(function(item) {{
                if (map.hasLayer(item.layer)) map.removeLayer(item.layer);
            }});
            renderRasterColorbars();
        }}

        function fetchTimeseries(folder, varName, dateCompact, latlng) {{
            var url = rasterClickServerUrl + '/timeseries?folder=' + encodeURIComponent(folder) +
                      '&var=' + encodeURIComponent(varName) +
                      '&date=' + encodeURIComponent(dateCompact) +
                      '&lat=' + encodeURIComponent(latlng.lat) +
                      '&lng=' + encodeURIComponent(latlng.lng);
            return fetch(url)
                .then(function(r) {{
                    if (!r || !r.ok) return null;
                    return r.text().then(function(t) {{
                        try {{ return JSON.parse(t); }} catch (e) {{ return null; }}
                    }});
                }})
                .catch(function() {{ return null; }});
        }}
        function fetchTimeseriesAllDays(folder, varName, latlng) {{
            var dates = timelineDatesUnique.length ? timelineDatesUnique : [currentDateCompact || ''];
            var promises = dates.map(function(dc) {{
                if (!dc) return Promise.resolve(null);
                return fetchTimeseries(folder, varName, dc, latlng);
            }});
            return Promise.all(promises).then(function(resps) {{
                var series = [];
                var offsets = [];
                var lengths = [];
                var labels = [];
                var hoursByDay = [];
                var cursor = 0;
                for (var i = 0; i < dates.length; i++) {{
                    var resp = resps[i];
                    var vals = (resp && resp.values) ? resp.values : [];
                    offsets.push(cursor);
                    lengths.push(vals.length);
                    labels.push(dates[i]);
                    hoursByDay.push(dayHoursMap[dates[i]] || []);
                    for (var j = 0; j < vals.length; j++) series.push(vals[j]);
                    cursor += vals.length;
                }}
                if (!series.length) return null;
                return {{ values: series, dayMeta: {{ offsets: offsets, lengths: lengths, labels: labels, hours: hoursByDay }} }};
            }});
        }}
        function showServerError() {{
            L.popup().setLatLng(map.getCenter()).setContent('Server error. Is raster_click_server running?').openOn(map);
        }}

        function renderTimeseriesSvg(values, w, h, unit, bands, dayMeta) {{
            w = w || 300; h = h || 110;
            var multiDay = dayMeta && dayMeta.labels && dayMeta.labels.length > 1;
            var padL = 40, padR = 6, padT = multiDay ? 16 : 6, padB = 16;
            var plotW = w - padL - padR;
            var plotH = h - padT - padB;
            var all = [];
            values.forEach(function(v) {{ if (v !== null && v !== undefined && !isNaN(v)) all.push(v); }});
            if (!all.length) return '';
            var minv = Math.min.apply(null, all);
            var maxv = Math.max.apply(null, all);
            if (minv === maxv) maxv = minv + 1e-6;
            var denom = (values.length > 1) ? (values.length - 1) : 1;
            function xForIndex(i) {{
                return padL + (i / denom) * plotW;
            }}
            var bandRects = '';
            if (bands && bands.length) {{
                var bandsSorted = bands.slice().sort(function(a, b) {{
                    var a0 = (a.range && a.range.length) ? a.range[0] : 0;
                    var b0 = (b.range && b.range.length) ? b.range[0] : 0;
                    return a0 - b0;
                }});
                bandsSorted.forEach(function(b) {{
                    var r0 = b.range ? b.range[0] : null;
                    var r1 = b.range ? b.range[1] : null;
                    if (r0 === null || r1 === null) return;
                    var c0 = Math.max(r0, minv);
                    var c1 = Math.min(r1, maxv);
                    if (!isFinite(c0) || !isFinite(c1)) return;
                    if (c1 <= minv || c0 >= maxv) return;
                    var y1 = padT + (1 - (c0 - minv) / (maxv - minv)) * plotH;
                    var y2 = padT + (1 - (c1 - minv) / (maxv - minv)) * plotH;
                    var yTop = Math.min(y1, y2);
                    var yH = Math.abs(y2 - y1);
                    bandRects += '<rect x=\"' + padL + '\" y=\"' + yTop.toFixed(2) + '\" width=\"' + plotW + '\" height=\"' + yH.toFixed(2) + '\" fill=\"' + (b.color || '#ccc') + '\" opacity=\"0.25\"/>';
                }});
            }}
            var svg = '<svg width=\"' + w + '\" height=\"' + h + '\" style=\"overflow:visible;\" xmlns=\"http://www.w3.org/2000/svg\">';
            svg += '<rect x=\"0\" y=\"0\" width=\"' + w + '\" height=\"' + h + '\" fill=\"#fff\" stroke=\"#eee\"/>' + bandRects;
            svg += '<line x1=\"' + padL + '\" y1=\"' + (padT + plotH) + '\" x2=\"' + (padL + plotW) + '\" y2=\"' + (padT + plotH) + '\" stroke=\"#666\" stroke-width=\"1\"/>';
            svg += '<line x1=\"' + padL + '\" y1=\"' + padT + '\" x2=\"' + padL + '\" y2=\"' + (padT + plotH) + '\" stroke=\"#666\" stroke-width=\"1\"/>';
            var ticksY = 4;
            var ticksX = [];
            if (!multiDay) {{
                var hours = (dayMeta && dayMeta.hours && dayMeta.hours.length) ? dayMeta.hours[0] : null;
                if (hours && hours.length) {{
                    var candidates = [0,6,12,18,23];
                    for (var ci = 0; ci < candidates.length; ci++) {{
                        var h = candidates[ci];
                        var idx = hours.indexOf(h);
                        if (idx >= 0) ticksX.push({{ idx: idx, label: pad2(h) }});
                    }}
                    if (!ticksX.length) {{
                        ticksX.push({{ idx: 0, label: pad2(hours[0]) }});
                        if (hours.length > 1) ticksX.push({{ idx: hours.length - 1, label: pad2(hours[hours.length - 1]) }});
                    }}
                }} else {{
                    ticksX = [0,6,12,18,23].map(function(h) {{ return {{ idx: h, label: pad2(h) }}; }});
                }}
            }}
            for (var t = 0; t <= ticksY; t++) {{
                var tv = minv + (t / ticksY) * (maxv - minv);
                var ty = padT + (1 - t / ticksY) * plotH;
                svg += '<line x1=\"' + padL + '\" y1=\"' + ty + '\" x2=\"' + (padL + plotW) + '\" y2=\"' + ty + '\" stroke=\"#eee\" stroke-width=\"1\"/>';
                svg += '<text x=\"' + (padL - 4) + '\" y=\"' + (ty + 3) + '\" text-anchor=\"end\" font-size=\"9\" fill=\"#333\">' +
                       tv.toFixed(1) + (unit ? (' ' + unit) : '') + '</text>';
            }}
            if (multiDay && dayMeta && dayMeta.offsets && dayMeta.lengths) {{
                for (var di = 0; di < dayMeta.offsets.length; di++) {{
                    var start = dayMeta.offsets[di] || 0;
                    var len = dayMeta.lengths[di] || 0;
                    if (len <= 0) continue;
                    if (di > 0) {{
                        var xb = xForIndex(start);
                        svg += '<line x1=\"' + xb.toFixed(2) + '\" y1=\"' + padT + '\" x2=\"' + xb.toFixed(2) + '\" y2=\"' + (padT + plotH) + '\" stroke=\"#ddd\" stroke-width=\"1\"/>';
                    }}
                    var mid = start + Math.max(0, (len - 1) / 2);
                    var xm = xForIndex(mid);
                    var lbl = labelDateCompact(dayMeta.labels[di] || '');
                    if (lbl) {{
                        svg += '<text x=\"' + xm.toFixed(2) + '\" y=\"' + (padT - 4) + '\" text-anchor=\"middle\" font-size=\"9\" fill=\"#333\">' + lbl + '</text>';
                    }}
                }}
            }}
            ticksX.forEach(function(tk) {{
                var tx = xForIndex(tk.idx);
                svg += '<line x1=\"' + tx + '\" y1=\"' + (padT + plotH) + '\" x2=\"' + tx + '\" y2=\"' + (padT + plotH + 4) + '\" stroke=\"#666\" stroke-width=\"1\"/>';
                svg += '<text x=\"' + tx + '\" y=\"' + (padT + plotH + 12) + '\" text-anchor=\"middle\" font-size=\"9\" fill=\"#333\">' + tk.label + '</text>';
            }});
            var path = '';
            var started = false;
            for (var i = 0; i < values.length; i++) {{
                var v = values[i];
                if (v === null || v === undefined || isNaN(v)) {{
                    started = false;
                    continue;
                }}
                var x = xForIndex(i);
                var t = (v - minv) / (maxv - minv);
                var y = padT + (1 - t) * plotH;
                if (!started) {{
                    path += (path ? ' M ' : 'M ') + x.toFixed(2) + ' ' + y.toFixed(2);
                    started = true;
                }} else {{
                    path += ' L ' + x.toFixed(2) + ' ' + y.toFixed(2);
                }}
            }}
            if (path) svg += '<path d=\"' + path + '\" fill=\"none\" stroke=\"#2c3e50\" stroke-width=\"1.5\"/>';
            svg += '</svg>';
            return svg;
        }}

        function bindClickPopup() {{
            if (rasterClickMode !== 'server') return;
            map.on('click', function(e) {{
                var activeHourly = null;
                for (var i = 0; i < hourlyRasterLayers.length; i++) {{
                    if (map.hasLayer(hourlyRasterLayers[i].layer)) {{ activeHourly = hourlyRasterLayers[i]; break; }}
                }}
                if (activeHourly) {{
                    fetchTimeseriesAllDays(activeHourly.scenario_folder || '.', activeHourly.var_name || '', e.latlng)
                        .then(function(resp) {{
                            if (!resp || !resp.values) {{ showServerError(); return; }}
                            var unit = activeHourly.unit || '';
                            var bands = (activeHourly.var_name === 'UTCI') ? utciBands : [];
                            var dayMeta = resp.dayMeta || null;
                            var dateInfo = '';
                            if (dayMeta && dayMeta.labels && dayMeta.labels.length) {{
                                var first = labelDateCompact(dayMeta.labels[0] || '');
                                var last = labelDateCompact(dayMeta.labels[dayMeta.labels.length - 1] || '');
                                dateInfo = first && last ? (first + ' → ' + last) : '';
                            }}
                            var html = '<div style=\"font-size:12px; min-width:280px;\">' +
                                       '<div style=\"font-weight:600; margin-bottom:4px;\">' + (activeHourly.name || 'Hourly') + (unit ? (' [' + unit + ']') : '') + '</div>' +
                                       '<div style=\"margin-bottom:4px;\">Periodo ' + (dateInfo || (timelineLabels[currentIndex] || (pad2(currentHour) + ':00'))) + '</div>' +
                                       renderTimeseriesSvg(resp.values, 300, 110, unit, bands, dayMeta) +
                                       '</div>';
                            L.popup().setLatLng(e.latlng).setContent(html).openOn(map);
                        }});
                    return;
                }}
            }});
        }}

        var slider = document.getElementById('hour-slider');
        if (slider) slider.addEventListener('input', function(e) {{ updateRasterHour(parseInt(e.target.value)); }});
        var opSlider = document.getElementById('opacity-slider');
        if (opSlider) {{
            opSlider.addEventListener('input', function(e) {{ updateRasterOpacity(parseFloat(e.target.value)); }});
            updateRasterOpacity(parseFloat(opSlider.value));
        }}
        var hideBtn = document.getElementById('raster-hide-all');
        if (hideBtn) hideBtn.addEventListener('click', function() {{ hideAllOverlays(); }});
        map.on('overlayadd', function() {{ updateRasterHour(currentHour); renderRasterColorbars(); }});
        map.on('overlayremove', function() {{ updateRasterHour(currentHour); renderRasterColorbars(); }});

        updateRasterHour({default_index});
        renderRasterColorbars();
        bindClickPopup();
    }});
    """
    m.get_root().script.add_child(Element(js))


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build interactive folium maps from a single city output folder.")
    parser.add_argument("--solweig-dir", type=str, required=True, help="Folder containing UTCI_YYYYMMDD.tif etc.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for HTML + PNG tiles.")
    parser.add_argument("--default-hour", type=int, default=12, help="Default hour shown in slider (0-23).")
    parser.add_argument("--overwrite-png", action="store_true", help="Overwrite existing PNGs instead of skipping.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging (only warnings/errors).",
    )
    parser.add_argument(
        "--raster-click-mode",
        choices=["server", "off"],
        default="server",
        help="Click popup mode: server (exact) or off.",
    )
    parser.add_argument(
        "--raster-click-server-url",
        type=str,
        default="http://localhost:8765",
        help="Base URL for raster click server.",
    )
    parser.add_argument(
        "--raster-click-folder",
        type=str,
        default=".",
        help="Folder value sent to raster click server (relative to SOLWEIG_DIR).",
    )
    parser.add_argument(
        "--raster-click-server-autostart",
        action="store_true",
        help="Start raster_click_server.py automatically (binds 127.0.0.1:8765).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Bundle Leaflet assets locally so the HTML works offline (no CDN).",
    )
    parser.add_argument(
        "--render-max-dim",
        type=int,
        default=DEFAULT_RENDER_MAX_DIM,
        help="Max raster dimension (px) for rendered PNGs; lower = faster.",
    )
    parser.add_argument(
        "--stats-max-dim",
        type=int,
        default=DEFAULT_STATS_MAX_DIM,
        help="Max raster dimension (px) for min/max stats; lower = faster.",
    )
    parser.add_argument(
        "--hourly-vars",
        type=str,
        default=",".join(DEFAULT_HOURLY_VARS.keys()),
        help="Comma-separated vars to include (e.g. UTCI or UTCI,Ta).",
    )
    args = parser.parse_args(argv)
    verbose = not bool(args.quiet)

    solweig_dir = Path(args.solweig_dir)
    output_dir = Path(args.output_dir)
    maps_dir = output_dir / "maps"
    assets_dir = output_dir / "assets"
    default_hour = int(np.clip(args.default_hour, 0, 23))
    render_max_dim = int(args.render_max_dim)
    stats_max_dim = int(args.stats_max_dim)
    hourly_vars = parse_hourly_vars(args.hourly_vars)

    _log(f"solweig_dir={solweig_dir}", verbose=verbose)
    _log(f"output_dir={output_dir}", verbose=verbose)
    _log(
        f"default_hour={default_hour} overwrite_png={bool(args.overwrite_png)} offline={bool(args.offline)} "
        f"render_max_dim={render_max_dim} stats_max_dim={stats_max_dim}",
        verbose=verbose,
    )
    _log(f"hourly_vars={','.join(hourly_vars.keys())}", verbose=verbose)
    _log(
        f"raster_click_mode={args.raster_click_mode} raster_click_server_url={args.raster_click_server_url} "
        f"raster_click_folder={args.raster_click_folder} autostart={bool(args.raster_click_server_autostart)}",
        verbose=verbose,
    )

    if args.raster_click_mode == "server" and args.raster_click_server_autostart:
        # Keep compatibility with the original layout where this script lived at
        # FORECAST_UTCI/interactive_maps.py.
        server_script = Path(__file__).resolve().parent.parent / "raster_click_server.py"
        if server_script.exists():
            cmd = [sys.executable, str(server_script), "--solweig-dir", str(solweig_dir)]
            subprocess.Popen(cmd)
            _log("Raster click server started on http://127.0.0.1:8765", verbose=verbose)
        else:
            print("[warn] raster_click_server.py not found; cannot autostart.")

    output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    restore_assets = None
    if args.offline:
        _log(f"Ensuring offline assets in {assets_dir}", verbose=verbose)
        ok = ensure_offline_assets(assets_dir)
        if ok:
            _log("Offline assets OK; configuring folium to use local JS/CSS.", verbose=verbose)
            restore_assets = configure_folium_offline_assets(html_dir=maps_dir, assets_dir=assets_dir)
        else:
            print("[warn] Offline assets incomplete; the HTML may require internet access.")

    t0 = time.perf_counter()
    try:
        m = build_raster_map(
            solweig_dir=solweig_dir,
            output_dir=output_dir,
            html_dir=maps_dir,
            default_hour=default_hour,
            overwrite_png=args.overwrite_png,
            raster_click_mode=args.raster_click_mode,
            raster_click_server_url=args.raster_click_server_url,
            raster_click_folder=args.raster_click_folder,
            verbose=verbose,
            render_max_dim=render_max_dim,
            stats_max_dim=stats_max_dim,
            hourly_vars=hourly_vars,
        )
        out_html = maps_dir / "interactive_map_raster.html"
        m.save(out_html)
        _log(f"Saved raster map: {out_html}", verbose=verbose)
    finally:
        if restore_assets is not None:
            restore_assets()
    _log(f"Done in {time.perf_counter() - t0:.2f}s", verbose=verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
