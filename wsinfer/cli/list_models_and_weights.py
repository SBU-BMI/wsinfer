import click

from ..modellib import models


@click.command()
def cli():
    """Show all available models and weights."""
    models_weights = models.list_all_models_and_weights()

    print("+------------------------------------------+")
    click.secho("| MODEL               WEIGHTS              |", bold=True)
    print("| ======================================== |")
    _prev_model = models_weights[0][0]
    for model_name, weights_name in models_weights:
        if _prev_model != model_name:
            print("| ---------------------------------------- |")
        _prev_model = model_name
        print(f"| {model_name:<20}{weights_name:<20} |")
    print("+------------------------------------------+")
