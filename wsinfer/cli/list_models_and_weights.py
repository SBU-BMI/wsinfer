import click

from ..modellib import models


@click.command()
def cli():
    """Show all available models and weights."""
    models_weights = models.list_all_models_and_weights()

    weights = [models.get_model_weights(*mw) for mw in models_weights]

    print("+-----------------------------------------------------------+")
    click.secho(
        "| MODEL             WEIGHTS        RESOLUTION               |", bold=True
    )
    print("| ========================================================= |")
    _prev_model = models_weights[0][0]
    for (model_name, weights_name), weight_obj in zip(models_weights, weights):
        if _prev_model != model_name:
            print("| --------------------------------------------------------- |")
        _prev_model = model_name
        r = f"{weight_obj.patch_size_pixels} px @ {weight_obj.spacing_um_px} um/px"
        print(f"| {model_name:<18}{weights_name:<15}{r:<25}|")
    print("+-----------------------------------------------------------+")
