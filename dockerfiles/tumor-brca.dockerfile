# Breast cancer tumor detection model (resnet34).
#
# Note about versioning: We should not use the 'latest' tag because it is a moving
# target. We should prefer using a versioned release of the wsi_inference pipeline.
FROM kaczmarj/patch-classification-pipeline:v0.2.1

# The CLI will use these env vars for model and weights.
ENV WSINFER_MODEL="resnet34"
ENV WSINFER_WEIGHTS="TCGA-BRCA-v1"

# Download the weights.
RUN python -c "from wsi_inference.modellib import models; models.$WSINFER_MODEL(\"$WSINFER_WEIGHTS\")" \
    # Downloaded models are mode 0600. Make them readable by all users.
    && chmod -R +r $TORCH_HOME/hub/checkpoints/
