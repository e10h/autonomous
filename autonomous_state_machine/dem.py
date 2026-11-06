"""Bounded-memory GeoTIFF ingestion; conservative gradients survive decimation."""

from dataclasses import asdict
import hashlib
import json
import math
import mmap
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .terrain import TraversabilityLimits


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_geotiff(
    source,
    output,
    resolution_m=None,
    vertical_scale=1.0,
    cliff_deg=22.0,
    native_tile_size=512,
    bounds=None,
):
    """Write memory-mappable prior, preserving native hard hazards in every bin.

    Input must be an axis-aligned projected raster. Horizontal CRS units must already be metres; vertical_scale must be explicitly chosen for input elevation units.
    Unknown pixels and cells whose gradients touch unknown pixels are blocked.
    `bounds`, when supplied, is in source CRS coordinates and crops without resampling.
    """
    import rasterio
    from rasterio.windows import Window, from_bounds

    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite terrain artifact: {output}")
    if not math.isfinite(vertical_scale) or vertical_scale <= 0:
        raise ValueError("Vertical scale must be positive")
    if native_tile_size < 4:
        raise ValueError("Tile size must be at least 4")
    limits = TraversabilityLimits(cliff_deg=cliff_deg)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".terrain-", dir=output.parent))
    try:
        with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024), rasterio.open(source) as ds:
            if ds.count != 1:
                raise ValueError("Expected a single-band elevation DEM")
            block_rows, block_cols = ds.block_shapes[0]
            if block_rows * block_cols * np.dtype(ds.dtypes[0]).itemsize > 64 * 1024 * 1024:
                raise ValueError("Source raster blocks exceed the ingestion memory budget; retile as a COG first")
            if ds.crs is None or not ds.crs.is_projected:
                raise ValueError(
                    "DEM must have a projected CRS; reproject geographic/unknown coordinates before ingestion"
                )
            if (
                ds.transform.b
                or ds.transform.d
                or ds.transform.a <= 0
                or ds.transform.e >= 0
            ):
                raise ValueError(
                    "DEM must be an axis-aligned north-up raster; reproject rotated or flipped rasters first"
                )
            units, unit_scale = ds.crs.linear_units_factor
            if not math.isclose(unit_scale, 1.0, rel_tol=1e-9):
                raise ValueError("DEM CRS must use metres; explicitly reproject feet-based coordinates before ingestion")
            rx, ry = ds.transform.a * unit_scale, -ds.transform.e * unit_scale
            if not math.isclose(rx, ry, rel_tol=1e-6):
                raise ValueError("DEM pixels must be square in metric units")
            if resolution_m is not None and (
                not math.isfinite(resolution_m) or resolution_m < rx - 1e-6
            ):
                raise ValueError(
                    "Planning resolution cannot invent finer detail than the native DEM"
                )
            factor = max(1, math.ceil((resolution_m or rx) / rx - 1e-9))
            row0, col0, height, width = 0, 0, ds.height, ds.width
            if bounds is not None:
                win = from_bounds(*bounds, transform=ds.transform)
                row0, col0 = math.floor(win.row_off), math.floor(win.col_off)
                endr, endc = math.ceil(win.row_off + win.height), math.ceil(
                    win.col_off + win.width
                )
                if row0 < 0 or col0 < 0 or endr > ds.height or endc > ds.width:
                    raise ValueError("Crop extends outside DEM")
                height, width = endr - row0, endc - col0
            if min(height, width) < 2:
                raise ValueError("DEM crop must contain at least two rows and columns")
            shape = (math.ceil(height / factor), math.ceil(width / factor))
            if min(shape) < 2:
                raise ValueError(
                    "Planning resolution leaves fewer than two rows/columns"
                )
            names = (
                "elevation",
                "gx_min",
                "gx_max",
                "gy_min",
                "gy_max",
                "slope_deg",
                "blocked",
            )
            arrays = {
                name: np.lib.format.open_memmap(
                    staging / (name + ".npy"),
                    mode="w+",
                    shape=shape,
                    dtype=np.bool_ if name == "blocked" else np.float32,
                )
                for name in names
            }
            tile = max(1, native_tile_size // factor)
            for r in range(0, shape[0], tile):
                for c in range(0, shape[1], tile):
                    nr, nc = min(tile, shape[0] - r), min(tile, shape[1] - c)
                    # Each tile has a native-pixel halo; gradients at tile seams
                    # are identical to gradients computed over the whole raster.
                    rr, cc = row0 + r * factor, col0 + c * factor
                    h, w = nr * factor, nc * factor
                    rs, cs = max(0, rr - 1), max(0, cc - 1)
                    re, ce = min(ds.height, rr + h + 1), min(ds.width, cc + w + 1)
                    raw = (
                        ds.read(1, window=Window(cs, rs, ce - cs, re - rs), masked=True)
                        .astype("float32")
                        .filled(np.nan)
                    )
                    raw *= vertical_scale
                    z = np.full((h + 2, w + 2), np.nan, dtype="float32")
                    z[rs - (rr - 1) : re - (rr - 1), cs - (cc - 1) : ce - (cc - 1)] = (
                        raw
                    )
                    gy, gx = np.gradient(z, -ry, rx)
                    gx = gx[1:-1, 1:-1]
                    gy = gy[1:-1, 1:-1]
                    z = z[1:-1, 1:-1]
                    # Block padded portions of incomplete coarse cells, too.
                    valid = np.isfinite(z) & np.isfinite(gx) & np.isfinite(gy)
                    ir, ic = np.ogrid[:h, :w]
                    valid &= (r * factor + ir < height) & (c * factor + ic < width)
                    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
                    hard = (~valid) | (slope > cliff_deg + 1e-6)
                    block = lambda x: x.reshape(nr, factor, nc, factor)
                    sl = np.s_[r : r + nr, c : c + nc]
                    arrays["blocked"][sl] = block(hard).any(axis=(1, 3))
                    for name, data, op in [
                        ("gx_min", gx, np.min),
                        ("gx_max", gx, np.max),
                        ("gy_min", gy, np.min),
                        ("gy_max", gy, np.max),
                        ("slope_deg", slope, np.max),
                        ("elevation", z, np.mean),
                    ]:
                        arrays[name][sl] = op(block(data), axis=(1, 3))
                # Flush each completed band of output rows and release mapped
                # pages; GDAL's input cache is explicitly bounded above.
                for a in arrays.values():
                    a.flush()
                    if hasattr(a._mmap, "madvise"):
                        a._mmap.madvise(mmap.MADV_DONTNEED)
            for a in arrays.values():
                a.flush()
            xcorner, ycorner = ds.transform @ (col0, row0)
            meta = {
                "schema": "yonder.terrain.v1",
                "source_name": source.name,
                "source_sha256": _sha256(source),
                "crs": ds.crs.to_string(),
                "crs_wkt": ds.crs.to_wkt(),
                "coordinate_unit": "metre",
                "source_horizontal_unit": units,
                "source_horizontal_to_m": unit_scale,
                "vertical_scale_to_m": vertical_scale,
                "native_resolution_m": rx,
                "resolution_m": rx * factor,
                "x0": xcorner * unit_scale + rx * factor / 2,
                "y0": ycorner * unit_scale - ry * factor / 2,
                "dx": rx * factor,
                "dy": -ry * factor,
                "shape": list(shape),
                "aggregation_factor": factor,
                "source_window": [row0, col0, height, width],
                "gradient_bounds": "conservative component intervals",
                "limits": asdict(limits),
                "unknown_policy": "blocked",
                "review_status": "unreviewed",
            }
            meta["terrain_id"] = hashlib.sha256(
                json.dumps(meta, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            (staging / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        arrays.clear()
        staging.rename(output)
        return meta
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
