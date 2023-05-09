:html_theme.sidebar_secondary.remove:

WSInfer: blazingly fast inference on whole slide images
=======================================================


.. image:: https://github.com/kaczmarj/wsinfer/actions/workflows/ci.yml/badge.svg
   :alt: GitHub workflow status
   :target: https://github.com/kaczmarj/wsinfer/actions/workflows/ci.yml
.. image:: https://readthedocs.org/projects/wsinfer/badge/?version=latest
  :alt: Documentation build status
  :target: https://wsinfer.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/pypi/v/wsinfer.svg
  :alt: PyPI version
  :target: https://pypi.org/project/wsinfer/
.. image:: https://img.shields.io/pypi/pyversions/wsinfer
  :alt: Supported Python versions
  :target: https://pypi.org/project/wsinfer/

|

🔥 🚀 **WSInfer** is a blazingly fast pipeline to run patch-based classification models
on whole slide images. It includes several built-in models for tumor and lymphocyte
detection, and it can be used with any PyTorch model as well. The built-in models
:ref:`are listed below <available-models>`.

Running inference on whole slide images is done with a single command line. For example,
this is the command used to generate the heatmap on this page.

::

   CUDA_VISIBLE_DEVICES=0 wsinfer run \
      --wsi-dir slides/ \
      --results-dir results/ \
      --model resnet34 \
      --weights TCGA-BRCA-v1 \
      --num-workers 8

To get started, please :ref:`install WSInfer<installing>` and check out the :ref:`User Guide`.
To get help, report issues or request features, please
`submit a new issue <https://github.com/SBU-BMI/wsinfer/issues/new>`_ on our GitHub
repository. If you would like to make your patch classification model available in WSInfer, please
get in touch with us! You can `submit a new GitHub issue <https://github.com/SBU-BMI/wsinfer/issues/new>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installing <installing>
   User Guide <user_guide>
   CLI <cli>


.. |img-tissue| image:: images/brca-tissue.png
  :alt: TCGA BRCA sample slide
.. |img-heatmap| image:: images/brca-heatmap.png
  :alt: Heatmap of breast cancer detection

+----------------+------------------------------+
| Original H&E   | Heatmap of Tumor Probability |
+================+==============================+
| |img-tissue|   | |img-heatmap|                |
+----------------+------------------------------+

.. _available-models:

Available models
----------------

.. list-table::
   :header-rows: 1

   * - Classification task
     - Output classes
     - Model
     - Weights
     - Resolution (px @ um/px)
     - Reference
   * - Breast adenocarcinoma detection
     - no-tumor, tumor
     - inception_v4
     - TCGA-BRCA-v1
     - 350 @ 0.25
     - `Ref <https://doi.org/10.1016%2Fj.ajpath.2020.03.012>`_
   * - Breast adenocarcinoma detection
     - no-tumor, tumor
     - resnet34
     - TCGA-BRCA-v1
     - 350 @ 0.25
     - `Ref <https://doi.org/10.1016%2Fj.ajpath.2020.03.012>`_
   * - Breast adenocarcinoma detection
     - no-tumor, tumor
     - vgg16mod
     - TCGA-BRCA-v1
     - 350 @ 0.25
     - `Ref <https://doi.org/10.1016%2Fj.ajpath.2020.03.012>`_
   * - Lung adenocarcinoma detection
     - lepidic, benign, acinar, micropapillary, mucinous, solid
     - resnet34
     - TCGA-LUAD-v1
     - 350 @ 0.5
     - `Ref <https://github.com/SBU-BMI/quip_lung_cancer_detection>`_
   * - Pancreatic adenocarcinoma detection
     - tumor-positive
     - preactresnet34
     - TCGA-PAAD-v1
     - 350 @ 1.5
     - `Ref <https://doi.org/10.1007/978-3-030-32239-7_60>`_
   * - Prostate adenocarcinoma detection
     - grade3, grade4or5, benign
     - resnet34
     - TCGA-PRAD-v1
     - 175 @ 0.5
     - `Ref <https://github.com/SBU-BMI/quip_prad_cancer_detection>`_
   * - Tumor-infiltrating lymphocyte detection
     - til-negative, til-positive
     - inception_v4nobn
     - TCGA-TILs-v1
     - 100 @ 0.5
     - `Ref <https://doi.org/10.3389/fonc.2021.806603>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
