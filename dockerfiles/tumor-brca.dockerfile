# Breast cancer tumor detection model (resnet34).
# TODO: change :dev tag to something else...
FROM kaczmarj/patch-classification-pipeline:dev
# TODO: Does singularity/apptainer mount over /tmp? If so, this will be a problem.
# TODO: YES, putting model weights in /tmp will not work... We need to use a
# writable directory that is not /tmp. Maybe make a new dir and chmod it.
# See the neurodocker dockerfile.
ENV TORCH_HOME=/tmp/torch
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@stonybrookmedicine.edu>"
# Download
RUN python -c 'from wsi_inference.modellib import models; models.resnet34("TCGA-BRCA-v1")'
