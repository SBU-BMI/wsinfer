# Breast cancer tumor detection model (resnet34).
#
# Note about versioning: We should not use the 'latest' tag because it is a moving
# target. We should prefer using a versioned release of the wsi_inference pipeline.
FROM kaczmarj/patch-classification-pipeline:v0.1.0
# Download the TCGA-BRCA-v1 weights for resnet34.
RUN python -c 'from wsi_inference.modellib import models; models.resnet34("TCGA-BRCA-v1")'
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@stonybrookmedicine.edu>"
