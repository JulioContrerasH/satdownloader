import pathlib
import torch
import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm
from satility.meta import MetaModel
from satility.ensemble import load_ensemble
from satility.geoutils import define_iteration, compute_valid_roi
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present



def load_model(
        ckpt_path: pathlib.Path | str | None = None,
        *, 
        mode: str = "none",
        device: str = "cpu"
    ) -> torch.nn.Module:


    if not ckpt_path:
        model = load_ensemble(
            mode=mode,
            device=device
        )
        return model
    else:
        ckpt_path = pathlib.Path(ckpt_path)
        state = torch.load(
            f=ckpt_path, 
            map_location=device, 
            weights_only=True
        )
        if "state_dict" in state:
            state = state["state_dict"]

        consume_prefix_in_state_dict_if_present(
            state_dict=state, 
            prefix="model."
        )

        ensemble = load_ensemble(
            mode="none",
            device=device
        )
        
        model = MetaModel(ensemble).to(device).eval() 

        model.load_state_dict(
            state_dict=state, 
            strict=True
        )
        model.requires_grad_(False)

    return model 


@torch.no_grad()
def predict(
    image_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    cloud_model: torch.nn.Module,
    chunk_size: int = 512,
    overlap: int = 32,
    device: str = "cpu"
) -> pathlib.Path:
    """
    Predict 'image_path' in overlapping patches of 'chunk_size' x 'chunk_size',
    but only write the valid (inner) region to avoid seam artifacts.

    This uses partial overlap logic:
      - For interior tiles, skip overlap//2 on each side.
      - For boundary tiles, we skip only the interior side to avoid losing data at the edges.

    Parameters
    ----------
    image_path : Path to input image.
    output_path : Path to output single-band mask.
    cloud_model : PyTorch model (already loaded with weights).
    chunk_size : Size of each tile to read from the source image (default 512).
    overlap : Overlap in pixels between adjacent tiles (default 32).
    device : "cpu" or "cuda:0".

    Returns
    -------
    pathlib.Path : The path to the created output image.
    """

    image_path = pathlib.Path(image_path)
    output_path = pathlib.Path(output_path)
    output_path_uncer = output_path.parent / (output_path.stem + "_uncer" + output_path.suffix)

    

    # 1) Validate metadata
    with rio.open(image_path) as src:
        meta = src.profile
        if not meta.get("tiled", False):
            raise ValueError("The input image is not marked as tiled in its metadata.")
        # Ensure the internal blocksize matches chunk_size
        if meta["blockxsize"] != chunk_size or meta["blockysize"] != chunk_size:
            raise ValueError(f"Image blocks must be {chunk_size}x{chunk_size}, "
                             f"got {meta['blockxsize']}x{meta['blockysize']}")
        height, width = meta["height"], meta["width"]

    # 2) Prepare output raster (1-band float)
    out_meta = meta.copy()
    out_meta["count"] = 1
    # Usually for a probability mask you'd use float32 for output
    out_meta["dtype"] = "float32"

    with rio.open(output_path, "w", **out_meta) as _:
        pass

    with rio.open(output_path_uncer, "w", **out_meta) as _:
        pass

    # 3) Get the list of tile offsets using define_iteration
    coords = define_iteration(
        dimension=(height, width),
        chunk_size=chunk_size,
        overlap=overlap
    )


    # 5) Iterate over tiles, read + partial-write
    with rio.open(image_path) as src, rio.open(output_path, "r+") as dst, rio.open(output_path_uncer, "r+") as dst_uncer:
        for idx, (row_off, col_off) in enumerate(tqdm(coords, desc="Inference")):
            

            # (a) Read a chunk_size x chunk_size window
            window = Window(
                col_off=col_off,
                row_off=row_off,
                width=chunk_size,
                height=chunk_size
            )
            # shape => (band=4, chunk_size, chunk_size)
            patch = src.read(window=window)

            # (b) Forward pass
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
            with torch.no_grad():
                response_model = cloud_model(patch_tensor) 
                if isinstance(response_model, tuple):
                    probs, uncer = response_model  # (1,1,H,W)
                else:
                    logits = response_model # (1,1,H,W)
                    probs = torch.sigmoid(logits)
                    uncer = None

            if uncer is not None:
                result = probs.unsqueeze(0).cpu().numpy()
                uncer_result = uncer.unsqueeze(0).cpu().numpy()  # shape (1, H, W)
            else:
                result = probs.squeeze(0).cpu().numpy() # (1, H, W)


            offset_x, offset_y, length_x, length_y, sub_x_start, sub_y_start = compute_valid_roi(
                row_off, col_off,
                chunk_size=chunk_size,
                overlap=overlap,
                height=height,
                width=width,
            )
          
            data = result[                                  
                :,
                sub_y_start : sub_y_start + length_y,
                sub_x_start : sub_x_start + length_x
            ]                                             
            if uncer is not None:
                mask = uncer_result[                                     
                    :,
                    sub_y_start : sub_y_start + length_y,
                    sub_x_start : sub_x_start + length_x
                ]                                    

            write_window = Window(
                offset_x,
                offset_y,
                length_x,
                length_y
            )

            dst.write(
                data, 
                window=write_window
            )

            if uncer is not None:
                dst_uncer.write(
                    mask, 
                    window=write_window
                )