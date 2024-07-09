# ![](docs/_static/logo.svg) WSInfer: deep learning inference on whole slide images

Original H&E                        |  Heatmap of Tumor Probability
:----------------------------------:|:-----------------------------------:
![](docs/_static/brca-tissue.png)  | ![](docs/_static/brca-heatmap.png)

🔥 🚀 Blazingly fast pipeline to run patch-based classification models on whole slide images.

[![Continuous Integration](https://github.com/SBU-BMI/wsinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/SBU-BMI/wsinfer/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/wsinfer/badge/?version=latest)](https://wsinfer.readthedocs.io/en/latest/?badge=latest)
[![Version on PyPI](https://img.shields.io/pypi/v/wsinfer.svg)](https://pypi.org/project/wsinfer/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/wsinfer)](https://pypi.org/project/wsinfer/)
[![Published in npj Precision Oncology](https://img.shields.io/badge/Published-npj_Precision_Oncology-blue)](https://doi.org/10.1038/s41698-024-00499-9)

> [!CAUTION]
> WSInfer is an academic project intended for research use only.

See https://wsinfer.readthedocs.io for documentation.

The main feature of WSInfer is a minimal command-line interface for running deep learning inference
on whole slide images. Here is an example:

```
wsinfer run \
   --wsi-dir slides/ \
   --results-dir results/ \
   --model breast-tumor-resnet34.tcga-brca
```

# Installation

WSInfer can be installed using `pip` or `conda`. WSInfer will install PyTorch automatically
if it is not installed, but this may not install GPU-enabled PyTorch even if a GPU is available.
For this reason, _install PyTorch before installing WSInfer_.

## Install PyTorch first

Please see [PyTorch's installation instructions](https://pytorch.org/get-started/locally/)
for help installing PyTorch. The installation instructions differ based on your operating system
and choice of `pip` or `conda`. Thankfully, the instructions provided
by PyTorch also install the appropriate version of CUDA. We refrain from including code
examples of installation commands because these commands can change over time. Please
refer to [PyTorch's installation instructions](https://pytorch.org/get-started/locally/)
for the most up-to-date instructions.

You will need a new-enough driver for your NVIDIA GPU. Please see
[this version compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility)
for the minimum versions required for different CUDA versions.

To test whether PyTorch can detect your GPU, check that this code snippet prints `True`.

```
python -c 'import torch; print(torch.cuda.is_available())'
```

## Install WSInfer

WSInfer can be installed with `pip` or `conda` (from `conda-forge`).

### Pip

To install the latest stable version, use

```
python -m pip install wsinfer
```

To install the _bleeding edge_ (which may have breaking changes), use

```
python -m pip install git+https://github.com/SBU-BMI/wsinfer.git
```

### Conda

To install the latest stable version, use

```
conda install -c conda-forge wsinfer
```

If you use `mamba`, simply replace `conda install` with `mamba install`.

### Developers

Clone this GitHub repository and install the package (in editable mode with the `dev` extras).

```
git clone https://github.com/SBU-BMI/wsinfer.git
cd wsinfer
python -m pip install --editable .[dev]
pre-commit install
```

We use `pre-commit` to automatically run various checks during `git commit`.

# Citation

If you find our work useful, please cite [our paper](https://doi.org/10.1038/s41698-024-00499-9)!

> Kaczmarzyk, J.R., O’Callaghan, A., Inglis, F. et al. Open and reusable deep learning for pathology with WSInfer and QuPath. *npj Precis. Onc.* **8**, 9 (2024). https://doi.org/10.1038/s41698-024-00499-9

```bibtex
@article{kaczmarzyk2024open,
  title={Open and reusable deep learning for pathology with WSInfer and QuPath},
  author={Kaczmarzyk, Jakub R. and O'Callaghan, Alan and Inglis, Fiona and Gat, Swarad and Kurc, Tahsin and Gupta, Rajarsi and Bremer, Erich and Bankhead, Peter and Saltz, Joel H.},
  journal={npj Precision Oncology},
  volume={8},
  number={1},
  pages={9},
  year={2024},
  month={Jan},
  day=10,
  doi={10.1038/s41698-024-00499-9},
  issn={2397-768X},
  url={https://doi.org/10.1038/s41698-024-00499-9}
}
```
