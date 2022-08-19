# Cancer segmentation pipelines for computational pathology

# Example

First, put all of your whole slide images into a directory. The name is arbitrary.
We will use `slides/`.

Create patches:

```
cd CLAM
python create_patches_fp.py --source ../slides/ --save_dir ../patch-results \
    --patch_size 340 --patch_spacing 0.25 --seg --patch --stitch --preset tcga.csv
cd ..
```

The patch coordinates are stored in `patch-results/patches` in an HDF5 file.

Run inference:

```
python run_inference.py  --wsi_dir slides/ --patch_dir patch-results/patches/ \
    --um_px 0.25 --patch_size 340 --model resnet34
```

This saves a CSV where each row represents a patch. Columns include the slide name,
patch location, and probabilties (in [0, 1]) for each class.
