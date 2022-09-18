# Breast cancer tumor detection model (resnet34).
# TODO: we should not use the latest tag, because it is a moving target.
FROM kaczmarj/patch-classification-pipeline:latest
# Download the TCGA-BRCA-v1 weights for resnet34.
RUN python -c 'from wsi_inference.modellib import models; models.resnet34("TCGA-BRCA-v1")'
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@stonybrookmedicine.edu>"
