FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /opt/wsinfer
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ git libopenslide0 \
    && python -m pip install --no-cache-dir --editable . \
    && rm -rf /var/lib/apt/lists/*
# Use a writable directory for downloading model weights. Default is ~/.cache, which is
# not guaranteed to be writable in a Docker container.
ENV TORCH_HOME=/var/lib/wsinfer
RUN mkdir -p "$TORCH_HOME" \
    && chmod 777 "$TORCH_HOME" \
    && chmod a+s "$TORCH_HOME"
WORKDIR /work
ENTRYPOINT ["wsinfer"]
CMD ["--help"]
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@stonybrookmedicine.edu>"
