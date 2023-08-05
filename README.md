# ![](docs/_static/logo.svg) WSInfer: deep learning inference on whole slide images

Original H&E                        |  Heatmap of Tumor Probability
:----------------------------------:|:-----------------------------------:
![](docs/_static/brca-tissue.png)  | ![](docs/_static/brca-heatmap.png)

🔥 🚀 Blazingly fast pipeline to run patch-based classification models on whole slide images.

[![Continuous Integration](https://github.com/SBU-BMI/wsinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/SBU-BMI/wsinfer/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/wsinfer/badge/?version=latest)](https://wsinfer.readthedocs.io/en/latest/?badge=latest)
[![Version on PyPI](https://img.shields.io/pypi/v/wsinfer.svg)](https://pypi.org/project/wsinfer/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/wsinfer)](https://pypi.org/project/wsinfer/)

See https://wsinfer.readthedocs.io for documentation.

# Installation

## Pip

WSInfer will install PyTorch automatically if it is not installed, but this may not
install GPU-enabled PyTorch even if a GPU is available. For this reason, install PyTorch
before installing WSInfer. Please see [PyTorch's installation instructions](https://pytorch.org/get-started/locally/)
for help install PyTorch.

```
python -m pip install wsinfer
```

To use the _bleeding edge_, use

```
python -m pip install git+https://github.com/SBU-BMI/wsinfer.git
```

## Developers

Clone this GitHub repository and install the package (in editable mode with the `dev` extras).

```
git clone https://github.com/SBU-BMI/wsinfer.git
cd wsinfer
python -m pip install --editable .[dev]
```

# Cutting a release

When ready to cut a new release, follow these steps:

1. Update the base image versions Dockerfiles in `dockerfiles/`. Update the version to
the version you will release.
2. Commit this change.
3. Create a tag, where VERSION is a string like `v0.3.6`:

    ```
    git tag -a -m 'wsinfer version VERSION' VERSION
    ```

4. Build wheel: `python -m build`
5. Create a fresh virtual environment and install the wheel. Make sure `wsinfer --help` works.
6. Push code to GitHub: `git push --tags`
6. Build and push docker images: `bash scripts/build_docker_images.sh 0.3.6 1`
7. Push wheel to PyPI: `twine upload dist/*`
