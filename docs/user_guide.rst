.. _User Guide:

User Guide
==========

This guide assumes that you have installed WSInfer. If you have not, please see :ref:`installing`.

It also assumes that you have a directory with at least one whole slide image.
If you do not, you can download a sample image from
https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/.

The rest of this page assumes that slides are saved to the directory :code:`slides`.

Getting help
------------

If you read the documentation but still have questions, need help, have feedback, found a bug,
or just want to chat, please
`submit a new issue <https://github.com/SBU-BMI/wsinfer/issues/new>`_ on our GitHub repo!

Get help on the command line
----------------------------

Most command line tools in macOS and Linux can help you with the :code:`--help` flag.
For example ::

   wsinfer --help

and ::

   wsinfer run --help

That will show you different subcommands, options, and expected inputs.

List available models
---------------------

WSInfer includes a Zoo of pretrained models. List them with the :code:`wsinfer-zoo` command line tool,
which is installed automatically with WSInfer. Please not the difference in the names :code:`wsinfer-zoo`
and :code:`wsinfer`.

::

   wsinfer-zoo ls


Run model inference
-------------------

The model inference workflow will separate each slide into patches and run model
inference on all patches. The results directory will include the model outputs,
patch coordinates, and metadata about the run.

To list available :code:`--model` options, use :code:`wsinfer-zoo ls`.

Here is an example of the minimum command-line for :code:`wsinfer run` (with arguments
:code:`--wsi-dir`, :code:`--results-dir`, and :code:`--model`).

::

   wsinfer run \
      --wsi-dir slides/ \
      --results-dir results/ \
      --model breast-tumor-resnet34.tcga-brca \

See :code:`wsinfer run --help` for more a list of options and how to use them.

The option :code:`--wsi-dir` is a directory containing only whole slide images. The option
:code:`--results-dir` is the path in which outputs are saved. The option :code:`--model`
is the name of a model available in WSInfer. The model weights and configuration are
downloaded from HuggingFace Hub. If you would like to use your own model, see :ref:`Use your own model`.

Outputs of model inference
--------------------------

The results directory will have several directories in it. We'll go over them now. ::

   results
   ├── masks
   │   ├── TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.jpg
   │   └── TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.jpg
   ├── model-outputs-csv
   │   ├── TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.csv
   │   └── TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.csv
   ├── model-outputs-geojson
   │   ├── TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.json
   │   └── TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.json
   ├── patches
   │   ├── TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.h5
   │   └── TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.h5
   └── run_metadata_20231110T235210.json

This hierarchy is inspired by CLAM's outputs. The :code:`masks` directory contains JPEG images
with thumbnails of the images and contours of the tissue and holes. The directory :code:`model-outputs-csv`
contains one CSV per slide, and each CSV contains the patchwise model outputs. Each row is a different patch.
Here are the feirst few rows of a sample CSV ::

   minx,miny,width,height,prob_Tumor
   4200,27300,2100,2100,6.4415544e-05
   4200,29400,2100,2100,9.763688e-05
   4200,31500,2100,2100,0.03654445

The directory :code:`model-outputs-geojson` contains the same information as the CSVs but in GeoJSON format.
GeoJSON is well-suited for spatial data, and QuPath can read it! Just drag and drop the GeoJSON file into the
QuPath window, and all of the patches and their model outputs will be appear. The directory :code:`patches`
contains HDF5 files of the patch coordinates. Last, there is a JSON file containing metadata about this run.
This has a timestamp in the filename in case you run inference multiple times to the same directory.

Run model inference in containers
---------------------------------

See https://hub.docker.com/r/kaczmarj/wsinfer/tags for all available containers.

The "base" image :code:`kaczmarj/wsinfer` includes
:code:`wsinfer` and all of its runtime dependencies. It does not, however, include
the downloaded model weights. Running a model will automatically download the weight,
but these weights will be removed once the container is stopped.

.. note::

  The image :code:`kaczmarj/wsinfer` does not include downloaded models. The models are downloaded
  automatically to :code:`~/.cache` but will be lost when the container is stopped if
  :code:`~/.cache` is not mounted.

Apptainer/Singularity
^^^^^^^^^^^^^^^^^^^^^

We use :code:`apptainer` in this example. You can replace that name with
:code:`singularity` if you do not have :code:`apptainer`.

Pull the container: ::

  apptainer pull docker://kaczmarj/wsinfer:latest

Run inference: ::

   apptainer run \
      --nv \
      --bind $(pwd) \
      --env CUDA_VISIBLE_DEVICES=0 \
      wsinfer_latest.sif run \
         --wsi-dir slides/ \
         --results-dir results/ \
         --model breast-tumor-resnet34.tcga-brca

Docker
^^^^^^

This requires Docker :code:`>=19.03` and the program :code:`nvidia-container-runtime-hook`. Please see the
`Docker documentation <https://docs.docker.com/config/containers/resource_constraints/#gpu>`_
for more information. If you do not have a GPU installed, you can use CPU by removing
:code:`--gpus all` from the command.

We use :code:`--user $(id -u):$(id -g)` to run the container as a non-root user (as ourself).
This way, the output files are owned by us. Without specifying this option, the output
files would be owned by the root user.

When mounting data, keep in mind that the workdir in the Docker container is :code:`/work`
(one can override this with :code:`--workdir`). Relative paths must be relative to the workdir.

One should mount their :code:`$HOME` directory onto the container. The registry of trained models
(a JSON file) is downloaded to :code:`~/.wsinfer-zoo-registry.json`, and trained models
are downloaded to :code:`~/.cache/huggingface/`.

.. note::

   Mount :code:`$HOME` into the container.

.. note::

  Using :code:`--num_workers > 0` will require a :code:`--shm-size > 256mb`.
  If the shm size is too low, a "bus error" will be thrown.

Pull the Docker image: ::

  docker pull kaczmarj/wsinfer:latest

Run inference: ::

   docker run --rm -it \
      --user $(id -u):$(id -g) \
      --mount type=bind,source=$HOME,target=$HOME \
      --mount type=bind,source=$(pwd),target=/work/ \
      --gpus all \
      --env CUDA_VISIBLE_DEVICES=0 \
      --env HOME=$HOME \
      --shm-size 512m \
      kaczmarj/wsinfer:latest run \
         --wsi-dir /work/slides/ \
         --results-dir /work/results/ \
         --model breast-tumor-resnet34.tcga-brca

.. _Use your own model:

Use your own model
------------------

WSInfer uses JSON configuration files to specify information required to run a patch classification model.

You can validate this configuration JSON file with ::

   wsinfer-zoo validate-config config.json

Once you create the configuration file, use the config with `wsinfer run`: ::

   wsinfer run --wsi-dir slides/ --results-dir results/ --model-path path/to/torchscript.pt --config config.json

Convert model outputs to Stony Brook format (QuIP)
--------------------------------------------------

The QuIP whole slide image viewer uses a particular format consisting of JSON and table files.

::

   wsinfer tosbu \
      --wsi-dir slides/ \
      --execution-id UNIQUE_ID_HERE \
      --study-id STUDY_ID_HERE \
      --make-color-text \
      --num-processes 16 \
      results/ \
      results/model-outputs-sbubmi/
