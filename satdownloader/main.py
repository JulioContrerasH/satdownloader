from __future__ import annotations

import pathlib
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from terracatalogueclient import Catalogue, Product, ProductFileType
from georeader.readers import probav_image_operational as probav
from georeader.readers import spotvgt_image_operational as spotvgt
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SensorCfg:
    collection_id: str
    reader_factory: Callable[[pathlib.Path], object]

SENSORS: dict[str, SensorCfg] = {
    "probav": SensorCfg(
        collection_id="urn:eop:VITO:PROBAV_L2A_1KM_HDF_V2",
        reader_factory=lambda p: probav.ProbaV(p, level_name="LEVEL2A"),
    ),
    "spot": SensorCfg(
        collection_id="urn:ogc:def:EOP:VITO:VGT_P",
        reader_factory=lambda p: spotvgt.SpotVGT(p),
    ),
}


# ─────────────────────────────── helper functions ────────────────────────────

def _runs_1d(vec: np.ndarray, *, min_len: int) -> List[Tuple[int, int]]:
    """Return (start, end) indices of True runs of length ≥ *min_len*."""
    runs: list[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, flag in enumerate(vec):
        if flag and not in_run:
            start, in_run = i, True
        elif not flag and in_run:
            if i - start >= min_len:
                runs.append((start, i))
            in_run = False
    if in_run and (len(vec) - start) >= min_len:
        runs.append((start, len(vec)))
    return runs


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2‑D (rows, cols), got {arr.shape}")
    return arr


def column_runs(mask, *, min_cols: int = 256) -> List[Tuple[int, int]]:
    """Contiguous valid‑column runs."""
    m = _ensure_2d(mask.values if hasattr(mask, "values") else np.asarray(mask))
    return _runs_1d(m.any(axis=0), min_len=min_cols)


def row_runs(mask, *, min_rows: int = 32) -> List[Tuple[int, int]]:
    """Contiguous valid‑row runs."""
    m = _ensure_2d(mask.values if hasattr(mask, "values") else np.asarray(mask))
    return _runs_1d(m.any(axis=1), min_len=min_rows)


def deep_find_lengths(data) -> List[int]:
    """Collect every 'length' value inside any nested structure."""
    if isinstance(data, dict):
        return sum((deep_find_lengths(v) for v in data.values()), []) + (
            [data["length"]] if "length" in data else []
        )
    if isinstance(data, list):
        return sum((deep_find_lengths(el) for el in data), [])
    return []


def write_strip(
    out_file: pathlib.Path,
    pixel_data: np.ndarray,
    valid_mask: np.ndarray,
    transform: Affine,
    crs: str | dict,
) -> None:
    """Save *pixel_data* (C, H, W) as a single‑strip COG."""
    bands, rows, cols = pixel_data.shape
    pixel_data = pixel_data.astype("float32", copy=False)
    pixel_data[:, ~valid_mask] = 0.0  # apply mask in‑place

    meta = dict(
        driver="GTiff",
        dtype="float32",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        interleave="band",
        compress="ZSTD",
        predictor=3,
        bigtiff="IF_SAFER",
        nodata=0.0,
        height=rows,
        width=cols,
        count=bands,
        crs=crs,
        transform=transform,
    )
    with rasterio.open(out_file, "w", **meta) as dst:
        dst.write(pixel_data)


# ───────────────────────────── processing pipeline ───────────────────────────

def process_product(
    prod: Product,
    outdir: pathlib.Path,
    sensor: str,
    *,
    cat: Catalogue | None = None,
    memory: None | int = None,
) -> None:
    

    start = prod.beginningDateTime.strftime("%Y-%m-%d")
    outdir = pathlib.Path(outdir)
    # date_folder = outdir /"HDF" / start
    date_folder = outdir / "HDF" /start
    date_folder.mkdir(exist_ok=True, parents=True)
    title = prod.title
    product_dir = date_folder / title


    if memory:
        if any(sz // 1024 ** 2 > memory for sz in deep_find_lengths(prod.geojson)):
            return

    try:
        cat.download_product(
            product=prod,
            path=date_folder,
            file_types=ProductFileType.DATA | ProductFileType.RELATED,
        )
    except Exception:
        return
    
    cfg = SENSORS[sensor]


    if sensor == "probav":
        
        hdf_path = product_dir / f"{title}.HDF5"

    elif sensor == "spot":
        change_directory = date_folder / "V003"
        hdf_path = product_dir
        change_directory.rename(hdf_path)
        


    reader = cfg.reader_factory(hdf_path)

    data = reader.load_radiometry()  # shape (4, H, W)
    mask = reader.load_mask()        # shape (H, W)

    # find valid rows / columns
    r_runs = row_runs(mask)
    c_runs = column_runs(mask)
    if not r_runs or not c_runs:
        return
    
    # find the first valid row
    y0, y1 = r_runs[0]
    parts = list(product_dir.parts)

    # Path to the GeoTIFF output
    parts[-3] = "GeoTIFF"
    out_dir = pathlib.Path(*parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove out_dir


    # Transform
    base_transform = reader.transform
    total = len(c_runs)

    for idx, (x0, x1) in enumerate(c_runs, 1):
        
        strip_basename = (
            f"{out_dir.name}_part{idx:02d}" if total > 1 else f"{out_dir.name}"
        )
        if sensor == "spot":
            out_dir.rmdir()
            out_file = out_dir.parent / f"{strip_basename}.tif"
        if sensor == "probav":
            out_file = out_dir / f"{strip_basename}.tif"

        if out_file.exists():
            continue

        strip = data.values[:, y0:y1, x0:x1]
        mstrip = mask.values[y0:y1, x0:x1]

        # Change transform to strip coordinates
        strip_transform = Affine.translation(x0 * base_transform.a, y0 * base_transform.e) * (
            base_transform
        )
        write_strip(out_file, strip, mstrip, strip_transform, reader.crs)



def download_image(
    catalogue: Catalogue,
    sensor: str,
    lon: float,
    lat: float,
    start: str,
    end: str,
    outdir: str
    ) -> None:
    
    cfg = SENSORS[sensor]
    products = catalogue.get_products(
        collection=cfg.collection_id,
        start=start,
        end=end,
        geometry=f"POINT({lon} {lat})",
    )


    for p in products:
        
        process_product(p, outdir, sensor, cat=catalogue)
        print(f"Downloaded {p.title}")