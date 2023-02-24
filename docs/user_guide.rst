.. _User Guide:

User Guide
==========

This guide assumes that you have a directory with at least one whole slide image.
If you do not, you can download a sample image from
https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/.

We assume the slides are saved to the directory :code:`slides`.

List available models
---------------------

::

   wsinfer list


Run model inference
-------------------

The model inference workflow will separate each slide into patches and run model
inference on all patches. The results directory will include the model outputs,
patch coordinates, and metadata about the run.

For available :code:`--model` and :code:`--weights` options, see :code:`wsinfer list`.

::

   CUDA_VISIBLE_DEVICES=0 wsinfer run \
      --wsi-dir slides/ \
      --results-dir results/ \
      --model resnet34 \
      --weights TCGA-BRCA-v1 \
      --num-workers 8

The option :code:`--num-workers` controls how many processes to use for data loading.
Using more workers will typically speed up processing at the expense of more RAM and
processor use.

The results directory includes a file named :code:`run_metadata.json` with provenance
information.

Run model inference in containers
---------------------------------

See https://hub.docker.com/r/kaczmarj/wsinfer/tags for all available containers.

The "base" image :code:`kaczmarj/wsinfer` includes
:code:`wsinfer` and all of its runtime dependencies. It does not, however, include
the downloaded model weights. Running a model will automatically download the weight,
but these weights will be removed once the container is stopped. For this reason, we
also provide model-specific containers. These are the same as the "base" image with the
addition of downloaded weights.

.. note::

  The base image :code:`kaczmarj/wsinfer` does not include downloaded models. The models
  will be downloaded automatically but will be lost when the container is stopped. Use
  versioned, application-specific containers like
  :code:`kaczmarj/wsinfer:0.3.2-tumor-brca` as they already include weights.

Apptainer/Singularity
^^^^^^^^^^^^^^^^^^^^^

We use :code:`apptainer` in this example. You can replace that name with
:code:`singularity` if you do not have :code:`apptainer`.

::

  apptainer pull docker://kaczmarj/wsinfer:0.3.2-tumor-paad
  CUDA_VISIBLE_DEVICES=0 apptainer run --nv --bind $(pwd) wsinfer_0.3.2-tumor-paad.sif \
    run --wsi-dir slides/ --results-dir results/

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

.. note::

  Using :code:`--num_workers > 0` will require a :code:`--shm-size > 256mb`.
  If the shm size is too low, a "bus error" will be thrown.

Pull the Docker image: ::

  docker pull kaczmarj/wsinfer:0.3.2-tumor-paad

Run inference: ::

  docker run --rm -it \
      --shm-size 512m \
      --gpus all \
      --env CUDA_VISIBLE_DEVICES=0 \
      --user $(id -u):$(id -g) \
      --mount type=bind,source=$(pwd),target=/work/ \
      kaczmarj/wsinfer:0.3.2-tumor-paad run \
          --wsi-dir slides/ \
          --results-dir results/ \
          --model resnet34 \
          --weights TCGA-BRCA-v1 \
          --num-workers 2


Use your own model
------------------

WSInfer uses YAML configuration files for models and weights. Please see the commented
example below.

.. code-block:: yaml

   # The version of the spec. At this time, only "1.0" is valid. (str)
   version: "1.0"
   # Models are referenced by the pair of (architecture, weights), so this pair must be unique.
   # The name of the architecture. We use timm to supply hundreds or network architectures,
   # so the name can be one of those models. If the architecture is not provided in timm,
   # then one can add an architecture themselves, but the code will have to be modified. (str)
   architecture: resnet34
   # A unique name for the weights for this architecture. (str)
   name: TCGA-BRCA-v1
   # Where to get the model weights. Either a URL or path to a file.
   # If using a URL, set the url_file_name (the name of the file when it is downloaded).
   # url: https://stonybrookmedicine.box.com/shared/static/dv5bxk6d15uhmcegs9lz6q70yrmwx96p.pt
   # url_file_name: resnet34-brca-20190613-01eaf604.pt
   # If not using a url, then 'file' must be supplied. Use an absolute or relative path. If
   # using a relative path, the path is relative to the location of the yaml file.
   file: path-to-weights.pt
   # Size of patches from the slides. (int)
   patch_size_pixels: 350
   # The microns per pixel of the patches. (float)
   spacing_um_px: 0.25
   # Number of output classes from the model. (int)
   num_classes: 2
   # Names of the model outputs. The order matters. class_names[0] is the name of the first
   # class of the model output.
   class_names:  # (list of strings)
      - notumor
      - tumor
   transform:
      # Size of images immediately prior to inputting to the model. (int)
      resize_size: 224
      # Mean and standard deviation for RGB values. (list of three floats)
      mean: [0.7238, 0.5716, 0.6779]
      std: [0.1120, 0.1459, 0.1089]

Once you create a configuration file, use the config with `wsinfer run`: ::

   wsinfer run --wsi-dir slides/ --results-dir results/ --config config.yaml


Convert model outputs to GeoJSON (QuPath)
-----------------------------------------

GeoJSON is a JSON format compatible with whole slide image viewers like QuPath.

::

   wsirun togeojson results/ geojson-results/

Convert model outputs to Stony Brook format (QuIP)
--------------------------------------------------

The QuIP whole slide image viewer uses a particular format consisting of JSON and table files.

::

   wsirun tosbu \
      --wsi-dir slides/ \
      --execution-id UNIQUE_ID_HERE \
      --study-id STUDY_ID_HERE \
      --make-color-text \
      --num-processes 16 \
      results/ \
      results/model-outputs-sbubmi/
