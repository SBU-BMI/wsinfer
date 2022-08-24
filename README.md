# Patch classification pipelines for computational pathology

Original H&E                        |  Heatmap of Tumor Probability
:----------------------------------:|:-----------------------------------:
![](sample-images/brca-tissue.png)  | ![](sample-images/brca-heatmap.png)

Run patch-based classification models on whole slide images of histology.

# Installation

Use the Docker / Singularity / Apptainer image, which includes all of the dependencies and scripts.

Alternatively, clone this repository and install the requirements.

```
git clone --recurse-submodules https://github.com/kaczmarj/patch-classification-pipeline.git
python -m venv venv
source ./venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt --find-links https://girder.github.io/large_image_wheels
```

# Example

Run a model on a directory of whole slide images. This command reads all of the images
in `brca-samples/` and writes results to `results/`.

```bash
CUDA_VISIBLE_DEVICES=2 singularity run --nv \
    --bind $PWD \
    --bind /data10:/data10:ro cancer-detection_latest.sif \
        --wsi_dir brca-samples/ \
        --results_dir results \
        --patch_size 340 \
        --um_px 0.25 \
        --model resnet34 \
        --num_classes 2 \
        --weights resnet34-jakub-state-dict-with-numbatchestracked.pt \
        --num_workers 8
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

## Convert to GeoJSON (for QuPath and other viewers)

```
python convert_csv_to_geojson.py results/model-outputs/TCGA-foobar.csv TCGA-foobar.json
```
