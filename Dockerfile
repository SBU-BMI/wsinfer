FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /opt/cancer-detection
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git libopenslide0 \
    && python -m pip install --no-cache-dir --requirement requirements.txt \
        --find-links https://girder.github.io/large_image_wheels \
    && apt-get autoremove --yes gcc \
    && rm -rf /var/lib/apt/lists/*
COPY . .
ENTRYPOINT ["python", "/opt/cancer-detection/run.py"]
