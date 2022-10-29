# Pancreatic adenocarcinoma detection model.
#
# Note about versioning: We should not use the 'latest' tag because it is a moving
# target. We should prefer using a versioned release of the wsi_inference pipeline.
FROM kaczmarj/patch-classification-pipeline:v0.2.0

# The CLI will use these env vars for model and weights.
ENV WSIRUN_MODEL="resnet34_preact"
ENV WSIRUN_WEIGHTS="TCGA-PAAD-v1"

# Download the weights.
RUN python -c "from wsi_inference.modellib import models; models.$WSIRUN_MODEL(\"$WSIRUN_WEIGHTS\")"
