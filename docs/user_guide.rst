User Guide
==========

.. _installation:

Installation
------------

To use WSInfer, first install it using pip:

.. code-block:: console

   pip install wsinfer --find-links https://girder.github.io/large_image_wheels

Examples
-------

List available models
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   wsinfer list

Run model inference
^^^^^^^^^^^^^^^^^^^

.. code-block:: console

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
