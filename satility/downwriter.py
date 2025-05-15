from __future__ import annotations

import pathlib
import numpy as np
import rasterio as rio
from rasterio.transform import Affine


def write_strip_image(
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
    with rio.open(out_file, "w", **meta) as dst:
        dst.write(pixel_data)

def write_strip_initial(
    out_file: pathlib.Path,
    pixel_data: np.ndarray,
    valid_mask: np.ndarray,
    transform: Affine,
    crs: str | dict,
) -> None:
    """Save *pixel_data* (C, H, W) as a single‑strip COG."""
    pixel_data = pixel_data[np.newaxis, ...]
    bands, rows, cols = pixel_data.shape
    pixel_data = pixel_data.astype("uint8", copy=False)
    pixel_data[:,~valid_mask] = 0  # apply mask in‑place

    meta = dict(
        driver="GTiff",
        dtype="uint8",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        interleave="band",
        compress="ZSTD",
        predictor=2,
        bigtiff="IF_SAFER",
        nodata=0,
        height=rows,
        width=cols,
        count=bands,
        crs=crs,
        transform=transform,
    )
    with rio.open(out_file, "w", **meta) as dst:
        dst.write(pixel_data)
