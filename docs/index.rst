:html_theme.sidebar_secondary.remove:

WSInfer: blazingly fast inference on whole slide images
=======================================================

🔥 🚀 **WSInfer** is a blazingly fast pipeline to run patch-based classification models
on whole slide images.

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
     - inceptionv4
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
     - grade3, grade4+5, benign
     - resnet34
     - TCGA-PRAD-v1
     - 175 @ 0.5
     - `Ref <https://github.com/SBU-BMI/quip_prad_cancer_detection>`_
   * - Tumor-infiltrating lymphocyte detection
     - til-negative, til-positive
     - inceptionv4nobn
     - TCGA-TILs-v1
     - 100 @ 0.5
     - `Ref <https://doi.org/10.3389/fonc.2021.806603>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
