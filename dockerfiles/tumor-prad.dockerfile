# Prostate adenocarcinoma detection model.
#
# Note about versioning: We should not use the 'latest' tag because it is a moving
# target. We should prefer using a versioned release of the wsinfer pipeline.
FROM kaczmarj/patch-classification-pipeline:v0.2.1

# The CLI will use these env vars for model and weights.
ENV WSINFER_MODEL="resnet34"
ENV WSINFER_WEIGHTS="TCGA-PRAD-v1"

# Download the weights.
RUN python -c "from wsinfer.modellib import models; models.$WSINFER_MODEL(\"$WSINFER_WEIGHTS\")" \
    # Downloaded models are mode 0600. Make them readable by all users.
    && chmod -R +r $TORCH_HOME/hub/checkpoints/
