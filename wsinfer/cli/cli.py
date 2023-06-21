import click

from .convert_csv_to_geojson import togeojson
from .convert_csv_to_sbubmi import tosbu
from .infer import run
from .patch import patch


# We use invoke_without_command=True so that 'wsinfer' on its own can be used for
# inference on slides.
@click.group()
@click.version_option()
def cli():
    """Run patch-level classification inference on whole slide images."""
    pass


cli.add_command(run)
cli.add_command(togeojson)
cli.add_command(tosbu)
cli.add_command(patch)
