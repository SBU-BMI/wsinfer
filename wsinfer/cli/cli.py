from __future__ import annotations

import logging
from typing import Literal

import click

from ..wsi import set_backend
from .convert_csv_to_geojson import togeojson
from .convert_csv_to_sbubmi import tosbu
from .infer import run
from .patch import patch

_logging_levels = ["debug", "info", "warning", "error", "critical"]


# We use invoke_without_command=True so that 'wsinfer' on its own can be used for
# inference on slides.
@click.group()
@click.option(
    "--backend",
    default=None,
    help="Backend for loading whole slide images.",
    type=click.Choice(["openslide", "tiffslide"]),
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(_logging_levels),
    help="Set the loudness of logging.",
)
@click.version_option()
def cli(backend: Literal["openslide"] | Literal["tiffslide"] | None, log_level: str):
    """Run patch-level classification inference on whole slide images."""

    # Configure logger.
    levels = {level: getattr(logging, level.upper()) for level in _logging_levels}
    level = levels[log_level]
    logging.basicConfig(level=level)

    if backend is not None:
        set_backend(backend)


cli.add_command(run)
cli.add_command(togeojson)
cli.add_command(tosbu)
cli.add_command(patch)
