.. _installing:

Installing and getting started
==============================

Prerequisites
-------------

WSInfer supports Python 3.8+ and has been tested on Windows, macOS, and Linux.

WSInfer can be installed using :code:`pip` or :code:`conda`. WSInfer will install PyTorch automatically
if it is not installed, but this may not install GPU-enabled PyTorch even if a GPU is available.
For this reason, *install PyTorch before installing WSInfer*.

Install PyTorch first
^^^^^^^^^^^^^^^^^^^^^

Please see `PyTorch's installation instructions <https://pytorch.org/get-started/locally/>`_
for help installing PyTorch. The installation instructions differ based on your operating system
and choice of :code:`pip` or :code:`conda`. Thankfully, the instructions provided
by PyTorch also install the appropriate version of CUDA. We refrain from including code
examples of installation commands because these commands can change over time. Please
refer to `PyTorch's installation instructions <https://pytorch.org/get-started/locally/>`_
for the most up-to-date instructions.

You will need a new-enough driver for your NVIDIA GPU. Please see
`this version compatibility table <https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility>`_
for the minimum versions required for different CUDA versions.

To test whether PyTorch can detect your GPU, check that this code snippet prints :code:`True` ::

    python -c 'import torch; print(torch.cuda.is_available())'

If your GPU is not available but you have a GPU, you can test if you installed a GPU-enabled PyTorch ::

    python -c 'import torch; print(torch.version.cuda)'

If that command does not print a version string (e.g., 11.7, 12.1), then you probably installed a CPU-only PyTorch.
Re-install PyTorch with CUDA support.

Another thing to test is that the environment variable :code:`CUDA_VISIBLE_DEVICES` is set. I (Jakub) have mine set to "0"
because I have one GPU on my machine. If it is set to something other than "0", then PyTorch will not be able to
detect the GPU.

Install WSInfer
----------------

WSInfer can be installed with :code:`pip` or :code:`conda` (from :code:`conda-forge`). In both cases, you get
the :code:`wsinfer` command line tool and Python package.

Pip
^^^

To install the latest stable version of WSInfer, use ::

    python -m pip install wsinfer

To check the installation, type ::

    wsinfer --help

To install the latest *unstable* version of WSInfer, use ::

    python -m pip install git+https://github.com/SBU-BMI/wsinfer

Conda
^^^^^

To install the latest stable version of WSInfer with :code:`conda`, use ::

    conda install -c conda-forge wsinfer

If you use :code:`mamba`, replace :code:`conda install` with :code:`mamba install`.

To check the installation, type ::

    wsinfer --help

Developers
^^^^^^^^^^

Clone the GitHub repository and install the package in editable mode with the :code:`dev` extras ::

    git clone https://github.com/SBU-BMI/wsinfer.git
    cd wsinfer
    python -m pip install --editable .[dev]


Supported slide backends
------------------------

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
