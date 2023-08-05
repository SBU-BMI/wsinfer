.. _installing:

Installing and getting started
==============================

Prerequisites
-------------

WSInfer supports Python 3.8+ and has been tested on Windows, macOS, and Linux.

WSInfer will install PyTorch automatically if it is not installed, but this may not
install GPU-enabled PyTorch even if a GPU is available. For this reason, install PyTorch
before installing WSInfer. Please see
`PyTorch's installation instructions <https://pytorch.org/get-started/locally/>`_.


.. note::

    Install PyTorch before installing WSInfer.


Install with pip
----------------

After having installed PyTorch, install the latest release of WSInfer from `PyPI <https://pypi.org/project/wsinfer/>`_. ::

    pip install wsinfer

This installs the :code:`wsinfer` Python package and the :code:`wsinfer` command line program. ::

    wsinfer --help

To install the latest unstable version of WSInfer, use ::

    pip install git+https://github.com/SBU-BMI/wsinfer

Supported backends
------------------

WSInfer supports two backends for reading whole slide images: `OpenSlide <https://openslide.org/>`_
and `TiffSlide <https://github.com/Bayer-Group/tiffslide>`_. When you install WSInfer, TiffSlide is also
installed. To install OpenSlide, install the compiled OpenSlide library and the Python package
:code:`openslide-python`. To choose the backend on the command line, use
:code:`wsinfer --backend=tiffslide ...` or :code:`wsinfer --backend=openslide ...`. In a Python script,
use :code:`wsinfer.wsi.set_backend`.

Containers
----------

See https://hub.docker.com/u/kaczmarj/wsinfer/ for available Docker images. These Docker images
can be used with Docker, Apptainer, or Singularity.

Docker:

::

    docker pull kaczmarj/wsinfer

Apptainer:

::

    apptainer pull docker://kaczmarj/wsinfer

Singularity:

::

    singularity pull docker://kaczmarj/wsinfer


Developers
----------

Clone the repository from https://github.com/SBU-BMI/wsinfer and install it in editable mode. ::

    git clone https://github.com/SBU-BMI/wsinfer.git
    cd wsinfer
    python -m pip install --editable .[dev,openslide]
    wsinfer --help

Getting started
---------------

The :code:`wsinfer` command line program is the main interface to WSInfer. Use the :code:`--help`
flag to show more information. ::

    wsinfer --help

To list the available trained models: ::

    wsinfer-zoo ls

To run inference on whole slide images: ::

    wsinfer run --wsi-dir slides/ --results-dir results/ --model breast-tumor-resnet34.tcga-brca

To convert model outputs to GeoJSON, for example to view in QuPath: ::

    wsinfer togeojson results/ model-outputs-geojson/
