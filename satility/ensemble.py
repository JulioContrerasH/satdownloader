import pathlib, mlstac, torch

def load_ensemble(
    model_dir: str | pathlib.Path | None = None,
    mode: str = "none",  
    device: str | torch.device = "cpu"
):
    if not model_dir:
        model_dir = pathlib.Path("PROBAandSPOT/ensemble")
    else:
        model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        mlstac.download(
            file="https://huggingface.co/tacofoundation/PROBAandSPOT/resolve/main/ensemble/mlm.json",
            output_dir=model_dir.as_posix(),
        )  
    ens = mlstac.load(model_dir.as_posix()).compiled_model(device=device, mode=mode)
        
    return ens
