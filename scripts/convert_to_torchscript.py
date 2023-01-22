"""Convert wsinfer models to torchscript."""

import hashlib
from pathlib import Path

import torch
from wsinfer import get_model_weights, list_all_models_and_weights


def sha256sum(path) -> str:
    """Calculate MD5 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 64)  # 64 kb
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def get_torchscript_frozen(arch: str, name: str):
    """Convert a wsinfer pytorch model to a frozen TorchScript model."""

    print("Loading model...")
    weights_obj = get_model_weights(architecture=arch, name=name)
    del arch, name

    width = weights_obj.patch_size_pixels
    example_input = torch.ones(1, 3, width, width)

    model = weights_obj.load_model()
    model.eval()

    print("Converting model to TorchScript...")
    try:
        model_jit = torch.jit.script(model, example_inputs=[(example_input,)])
    except RuntimeError:
        print("!!!!!!")
        print("torch.jit.script failed!")
    try:
        model_jit = torch.jit.trace(model, example_input)
    except Exception:
        return None, None
    model_jit_frozen = torch.jit.freeze(model_jit)

    print("Validating that jit model gives same output as original...")
    with torch.no_grad():
        output: torch.Tensor = model(example_input)
        output_jit: torch.Tensor = model_jit(example_input)
        assert torch.allclose(output, output_jit)
        output_jit_frozen: torch.Tensor = model_jit_frozen(example_input)
        assert torch.allclose(output, output_jit_frozen)

    return model_jit_frozen, weights_obj


if __name__ == "__main__":
    outdir = Path("jit-frozen-models")
    outdir.mkdir(exist_ok=True)

    for model, weights in list_all_models_and_weights():
        print(f"{model} with {weights} weights")
        frozen_model, w = get_torchscript_frozen(model, weights)
        if frozen_model is None:
            print("Failed to convert.")
            print("-" * 40)
            continue
        p = Path(w.url_file_name)
        p = outdir / f"{p.stem}-jit-frozen.pt"
        print(f"Saving to {p}")
        torch.jit.save(frozen_model, p)
        print("Calculating sha256 hash")
        sha = sha256sum(p)
        newname = p.parent / f"{p.stem}-{sha[:8]}.pt"
        print(f"Renaming to {newname}")
        p.rename(newname)
        print("-" * 40)
