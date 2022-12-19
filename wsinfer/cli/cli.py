import click

from .. import __version__


# We use invoke_without_command=True so that 'wsinfer' on its own can be used for
# inference on slides.
@click.group(invoke_without_command=True)
@click.version_option(__version__)
def cli():
    pass
