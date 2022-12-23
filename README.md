# Patch classification pipelines for computational pathology

Original H&E                        |  Heatmap of Tumor Probability
:----------------------------------:|:-----------------------------------:
![](sample-images/brca-tissue.png)  | ![](sample-images/brca-heatmap.png)

🔥 🚀 Blazingly fast pipeline to run patch-based classification models on whole slide images.

![Continuous Integration](https://github.com/kaczmarj/patch-classification-pipeline/actions/workflows/ci.yml/badge.svg)

# Table of contents

- [Available models](#available-models)
- [Installation](#installation)
  * [Pip](#pip)
  * [Containers](#containers)
    + [Containers for different classification tasks](#containers-for-different-classification-tasks)
  * [Developers](#developers)
- [Examples](#examples)
  * [Setup directories and data](#setup-directories-and-data)
  * [On "bare metal" (not inside a container)](#on-bare-metal-not-inside-a-container)
  * [Run in an Apptainer container (formerly Singularity)](#run-in-an-apptainer-container-formerly-singularity)
  * [Run in a Docker container](#run-in-a-docker-container)
  * [Output](#output)
  * [Convert to GeoJSON (for QuPath and other viewers)](#convert-to-geojson-for-qupath-and-other-viewers)
  * [Convert to Stony Brook QuIP format](#convert-to-stony-brook-quip-format)

# Available models

| Classification task                     | Output classes                                           | Model           | Weights name | Reference                                                    |
|-----------------------------------------|----------------------------------------------------------|-----------------|--------------|--------------------------------------------------------------|
| Breast adenocarcinoma detection         | no-tumor, tumor                                          | inceptionv4     | TCGA-BRCA-v1 | [ref](https://doi.org/10.1016%2Fj.ajpath.2020.03.012)        |
| Breast adenocarcinoma detection         | no-tumor, tumor                                          | resnet34        | TCGA-BRCA-v1 | [ref](https://doi.org/10.1016%2Fj.ajpath.2020.03.012)        |
| Breast adenocarcinoma detection         | no-tumor, tumor                                          | vgg16_modified  | TCGA-BRCA-v1 | [ref](https://doi.org/10.1016%2Fj.ajpath.2020.03.012)        |
| Lung adenocarcinoma detection           | lepidic, benign, acinar, micropapillary, mucinous, solid | resnet34        | TCGA-LUAD-v1 | [ref](https://github.com/SBU-BMI/quip_lung_cancer_detection) |
| Pancreatic adenocarcinoma detection     | tumor-positive                                           | resnet34_preact | TCGA-PAAD-v1 | [ref](https://doi.org/10.1007/978-3-030-32239-7_60)          |
| Prostate adenocarcinoma detection       | grade3, grade4+5, benign                                 | resnet34        | TCGA-PRAD-v1 | [ref](https://github.com/SBU-BMI/quip_prad_cancer_detection) |
| Tumor-infiltrating lymphocyte detection | til-negative, til-positive                                             | inceptionv4     | TCGA-TILs-v1 | [ref](https://doi.org/10.3389/fonc.2021.806603)              |

# Installation

## Pip

Pip install this package from GitHub. First install `torch` and `torchvision`
(please see [the PyTorch documentation](https://pytorch.org/get-started/locally/)).
We do not install these dependencies automatically because their installation can vary based
on a user's system. Then use the command below to install this package.

```
python -m pip install --find-links https://girder.github.io/large_image_wheels wsinfer
```

To use the _bleed edge_, use

```
python -m pip install \
    --find-links https://girder.github.io/large_image_wheels \
    git+https://github.com/kaczmarj/patch-classification-pipeline.git
```

## Containers

Use the Docker / Singularity / Apptainer image, which includes all of the dependencies and scripts.
See [DockerHub](https://hub.docker.com/r/kaczmarj/patch-classification-pipeline/tags) for
the available tags.

- Apptainer / Singularity

    Replace apptainer with singularity if you do not have apptainer

    ```
    apptainer pull docker://kaczmarj/patch-classification-pipeline
    ```

- Docker

    ```
    docker pull kaczmarj/patch-classification-pipeline
    ```

### Containers for different classification tasks

We distribute containers that include weights for different tasks, and these containers
have a simplified command-line interface of `command SLIDE_DIR OUTPUT_DIR`.
See [DockerHub](https://hub.docker.com/r/kaczmarj/patch-classification-pipeline/tags) for
the available tags. The Dockerfiles are in [`dockerfiles/`](/dockerfiles/) Here is an example:

```
apptainer pull docker://kaczmarj/patch-classification-pipeline:v0.2.0-paad-resnet34
CUDA_VISIBLE_DEVICES=0 apptainer run --nv --bind $(pwd) patch-classification-pipeline_v0.2.0-paad-resnet34.sif \
    --wsi-dir slides/ --results-dir results/
```

## Developers

Clone this GitHub repository and install the package (in editable mode with the `dev` extras).

```
git clone https://github.com/kaczmarj/patch-classification-pipeline.git
cd patch-classification-pipeline
python -m pip install --editable .[dev] --find-links https://girder.github.io/large_image_wheels
```

# Examples

Here we demonstrate running this pipeline on a sample image. Before going through this,
please install the package (see [Installation](#installation)).

## Setup directories and data

We make a new directory to store this example, including data and results. Enter the
following commands into a terminal. This will download a sample whole slide image
(170 MB). For this example, we only use one whole slide image, but you can apply this
pipeline to an arbitrary number of whole slide images &mdash; simply put them all in the
same directory.

```
mkdir -p example-wsi-inference
cd example-wsi-inference
mkdir -p sample-images
cd sample-images
wget -nc https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs
cd ..
```

## List available models and weights

We use "model" as in architecture (like "resnet50"), and "weights" are the pretrained
parameters that are loaded into the model for a particular task (like "TCGA-BRCA-v1"
for breast cancer tumor detection). Use the following command to list all available
models and weights.

```
wsinfer list
```


## On "bare metal" (not inside a container)

Run the pipeline (without a container). This will apply the pipeline to all of the
images in `sample-images/` (only 1 in this example) and will write results to
`results/`. We set `CUDA_VISIBLE_DEVICES=0` to use the first GPU listed in
`nvidia-smi`. If you do not have a GPU, model inference can take about 20 minutes.

```
CUDA_VISIBLE_DEVICES=0 wsinfer run \
    --wsi-dir sample-images/ \
    --results-dir results/ \
    --model resnet34 \
    --weights TCGA-BRCA-v1 \
    --num-workers 8
```

## Run in an Apptainer container (formerly Singularity)

I use the commands `apptainer` here, but if you don't have `apptainer`, you can
replace that with `singularity`. The command line interfaces are the same (as of August 26, 2022).

```
apptainer pull docker://kaczmarj/patch-classification-pipeline
```

Run the pipeline in Apptainer.

```
CUDA_VISIBLE_DEVICES=0 apptainer run \
    --nv \
    --bind $(pwd) \
    --pwd $(pwd) \
    patch-classification-pipeline_latest.sif run \
        --wsi-dir sample-images/ \
        --results-dir results/ \
        --model resnet34 \
        --weights TCGA-BRCA-v1 \
        --num-workers 8
```

## Run in a Docker container

First, pull the Docker image.

```
docker pull kaczmarj/patch-classification-pipeline
```

This requires Docker `>=19.03` and the program `nvidia-container-runtime-hook`. Please see the
[Docker documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu)
for more information. If you do not have a GPU installed, you can use CPU by removing
`--gpus all` from the command below.

We use `--user $(id -u):$(id -g)` to run the container as a non-root user (as ourself).
This way, the output files are owned by us. Without specifying this option, the output
files would be owned by the root user.

When mounting data, keep in mind that the workdir in the Docker container is `/work`
(one can override this with `--workdir`). Relative paths must be relative to the workdir.

Note: using `--num_workers > 0` will require a `--shm-size > 256mb`. If the shm size is
too low, a "bus error" will be thrown.

```
docker run --rm -it \
    --shm-size 512m \
    --gpus all \
    --env CUDA_VISIBLE_DEVICES=0 \
    --user $(id -u):$(id -g) \
    --mount type=bind,source=$(pwd),target=/work/ \
    kaczmarj/patch-classification-pipeline run \
        --wsi-dir sample-images/ \
        --results-dir results/ \
        --model resnet34 \
        --weights TCGA-BRCA-v1 \
        --num-workers 2
```

## Output

This will create the following directory structure

```
results/
├── masks
├── model-outputs
├── patches
└── stitches
```

- masks contains PNGs of tissue masks
- model-outputs contains CSVs of model outputs
- patches contains HDF5 files of patch coordinates
- stitches contains PNGs with patches stitched together

The output also contains a file `results/run_metadata.json` containing metadata about the run.

## Convert to GeoJSON (for QuPath and other viewers)

GeoJSON is a standardized format to represent geometry. The results of model inference
are a type of geometric data structure. Popular whole slide image viewers like QuPath
are able to load labels in GeoJSON format.

```bash
wsirun togeojson results/ geojson-results
```

## Convert to Stony Brook QuIP format

The Stony Brook QuIP format uses a combination of JSON and plain text files. Provide
a unique `--execution-id` that identifies this run. An example could be `tcga-brca-resnet34-tumor`.
Also provide a `--study-id`, which could be `TCGA-BRCA`. The option `--make-color-text` will
create the `color-*` files that contain color information for each patch in the input slides.
This option is disabled by default because it adds significant processing time.

```bash
wsirun tosbu \
    --wsi-dir slides/ \
    --execution-id UNIQUE_ID_HERE \
    --study-id TCGA-BRCA \
    --make-color-text \
    --num-processes 16 \
    results/ \
    results/model-outputs-sbubmi/
```
