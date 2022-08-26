FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /opt/wsi-classification
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git libopenslide0 \
    && python -m pip install --no-cache-dir --editable . \
        --find-links https://girder.github.io/large_image_wheels \
    && apt-get autoremove --yes gcc \
    && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["wsi_run"]
