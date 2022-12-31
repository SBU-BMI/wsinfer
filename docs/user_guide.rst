User Guide
==========

List available models
---------------------

::

   wsinfer list

As of January 1, 2022, the output is ::

   +------------------------------------------+
   | MODEL               WEIGHTS              |
   | ======================================== |
   | inceptionv4         TCGA-BRCA-v1         |
   | ---------------------------------------- |
   | inceptionv4nobn     TCGA-TILs-v1         |
   | ---------------------------------------- |
   | preactresnet34      TCGA-PAAD-v1         |
   | ---------------------------------------- |
   | resnet34            TCGA-BRCA-v1         |
   | resnet34            TCGA-LUAD-v1         |
   | resnet34            TCGA-PRAD-v1         |
   | ---------------------------------------- |
   | vgg16mod            TCGA-BRCA-v1         |
   +------------------------------------------+


Run model inference
-------------------

The model inference workflow will separate each slide into patches and run model
inference on all patches. The results directory will include the model outputs,
patch coordinates, and metadata about the run.

For available :code:`--model` and :code:`--weights` options, see :code:`wsinfer list`.

::

   mkdir -p example-wsi-inference/sample-images
   cd example-wsi-inference/sample-images
   # Download a sample slide.
   wget -nc https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs
   cd ..
   # Run inference on the slide.
   CUDA_VISIBLE_DEVICES=0 wsinfer run \
      --wsi-dir sample-images/ \
      --results-dir results/ \
      --model resnet34 \
      --weights TCGA-BRCA-v1 \
      --num-workers 8


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
