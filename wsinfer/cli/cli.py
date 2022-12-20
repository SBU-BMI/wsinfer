import click

# from .. import __version__
from .convert_csv_to_geojson import cli as _cli_convert_to_geojson
from .convert_csv_to_sbubmi import cli as _cli_convert_to_sbubmi
from .infer import cli as _cli_inference


# We use invoke_without_command=True so that 'wsinfer' on its own can be used for
# inference on slides.
@click.group()
@click.version_option()
def cli():
    """Run patch-level classification inference on whole slide images."""
    pass


cli.add_command(_cli_inference, name="run")
cli.add_command(_cli_convert_to_geojson, name="togeojson")
cli.add_command(_cli_convert_to_sbubmi, name="tosbu")
