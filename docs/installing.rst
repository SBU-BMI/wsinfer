Installing and getting started
==============================

Prerequisites
-------------

WSInfer supports Python 3.7+.

Install PyTorch before installing WSInfer. Please see
`PyTorch's installation instructions <https://pytorch.org/get-started/locally/>`_.
WSInfer does not install PyTorch automatically because the installation depends on
the type of hardware a user has.

Manual installation
-------------------

After having installed PyTorch, install releases of WSInfer from `PyPI <https://pypi.org/project/wsinfer/>`_.
Be sure to include the line :code:`--find-links https://girder.github.io/large_image_wheels` to ensure
dependencies are installed properly. ::

    pip install wsinfer --find-links https://girder.github.io/large_image_wheels

This installs the :code:`wsinfer` Python package and the :code:`wsinfer` command line program. ::

    wsinfer --help

Containers
----------

See https://hub.docker.com/u/kaczmarj/wsinfer/ for available Docker images. ::

    docker pull kaczmarj/wsinfer

The main wsinfer container includes all dependencies and can run all models. There are also model-specific
Docker images that are more suitable for inference because those include pre-downloaded model weights. If using
the "base" WSInfer Docker image, model weights need to be downloaded every time the container
is used for inference, and the downloaded weights do not persist after the container is stopped.

Development installation
------------------------

Clone the repository from https://github.com/kaczmarj/wsinfer and install it in editable mode. ::

    git clone https://github.com/kaczmarj/wsinfer.git
    cd wsinfer
    pip install --editable .[dev]
    wsinfer --help

Getting started
---------------

The :code:`wsinfer` command line program is the main interface to WSInfer. Use the :code:`--help`
flag to show more information. ::

    wsinfer --help

To list the available trained models: ::

    wsinfer list

To run inference on whole slide images: ::

    wsinfer run --wsi-dir slides/ --results-dir results/ --model resnet34 --weights TCGA-BRCA-v1

To convert model outputs to GeoJSON, for example to view in QuPath: ::

    wsinfer togeojson results/ model-outputs-geojson/
