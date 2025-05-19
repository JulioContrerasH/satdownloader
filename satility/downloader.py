from __future__ import annotations

import pathlib
from satility.downutils import row_runs, column_runs, deep_find_lengths
from satility.downclass import SENSORS
from satility.downwriter import write_strip_image, write_strip_initial
from rasterio.transform import Affine
from terracatalogueclient import Catalogue, Product, ProductFileType

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
    initial_mask = reader.load_sm_cloud_mask()

    # find valid rows / columns
    r_runs = row_runs(mask)
    c_runs = column_runs(mask)
    if not r_runs or not c_runs:
        return
    
    # find the first valid row
    y0, y1 = r_runs[0]

    parts = list(product_dir.parts)

    
    out_dir = pathlib.Path(*(parts[:1] + ["GeotTiFF"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Transform
    base_transform = reader.transform
    total = len(c_runs)

    for idx, (x0, x1) in enumerate(c_runs, 1):
          
        out_file = pathlib.Path(
            f"{out_dir / title}_part{idx:02d}.tif" if total > 1 else f"{out_dir / (title +'.tif')}"
        )
        out_file_initial = pathlib.Path(
            f"{out_dir / title}_part{idx:02d}_initial_mask.tif" if total > 1 else f"{out_dir / (title +'_initial_mask.tif')}"
        )

        out_file_valid = pathlib.Path(
            f"{out_dir / title}_part{idx:02d}_valid_mask.tif" if total > 1 else f"{out_dir / (title +'_valid_mask.tif')}"
        )
        
        if out_file.exists() and out_file_initial.exists():
            continue

        strip = data.values[:, y0:y1, x0:x1]
        strip_initial = initial_mask.values[y0:y1, x0:x1]
        strip_valid = mask.values[y0:y1, x0:x1]
        mstrip = mask.values[y0:y1, x0:x1]

        # Change transform to strip coordinates
        strip_transform = Affine.translation(x0 * base_transform.a, y0 * base_transform.e) * (
            base_transform
        )
        write_strip_image(
            out_file = out_file, 
            pixel_data = strip, 
            valid_mask = mstrip, 
            transform = strip_transform, 
            crs = reader.crs
        )
        write_strip_initial(
            out_file = out_file_initial, 
            pixel_data = strip_initial, 
            valid_mask = mstrip, 
            transform = strip_transform, 
            crs = reader.crs
        )
        write_strip_initial(
            out_file = out_file_valid, 
            pixel_data = strip_valid, 
            valid_mask = mstrip, 
            transform = strip_transform, 
            crs = reader.crs
        )


def download_image(
    catalogue: Catalogue,
    sensor: str,
    start: str,
    end: str,
    *,
    outdir: str = "data",
    lon: float | None = None,
    lat: float | None = None,
    ) -> None:
    
    cfg = SENSORS[sensor]

    if (not lon is None) and (not lat is None): 
        products = catalogue.get_products(
            collection=cfg.collection_id,
            start=start,
            end=end,
            geometry=f"POINT({lon} {lat})",
        )
    else:
        products = catalogue.get_products(
            collection=cfg.collection_id,
            start=start,
            end=end,
        )

    for p in products:
        
        process_product(prod = p, 
                        outdir = outdir, 
                        sensor = sensor,
                        cat=catalogue
                        )
        print(f"Downloaded {p.title}")
