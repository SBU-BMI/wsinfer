# Cancer segmentation pipelines for computational pathology

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
