# Lung adenocarcinoma detection model.
#
# Note about versioning: We should not use the 'latest' tag because it is a moving
# target. We should prefer using a versioned release of the wsinfer pipeline.
FROM kaczmarj/wsinfer:0.3.2

# The CLI will use these env vars for model and weights.
ENV WSINFER_MODEL="resnet34"
ENV WSINFER_WEIGHTS="TCGA-LUAD-v1"

# Download the weights.
RUN python -c "from wsinfer.modellib.models import get_model_weights; get_model_weights(architecture=\"$WSINFER_MODEL\", name=\"$WSINFER_WEIGHTS\").load_model()" \
    # Downloaded models are mode 0600. Make them readable by all users.
    && chmod -R +r $TORCH_HOME/hub/checkpoints/
