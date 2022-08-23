# Patch classification pipelines for computational pathology

Run patch-based classification models on whole slide images of histology.

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
