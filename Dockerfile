# FIXME: when using the torch 2.0.1 image, we get an error
#   OSError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0
# The error is fixed by
#   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=Etc/UTC

# Install system dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ git libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

# Install wsinfer.
WORKDIR /opt/wsinfer
COPY . .
RUN python -m pip install --no-cache-dir --editable . openslide-python tiffslide

# Test that the program runs (and also download the registry JSON file).
RUN wsinfer --help

WORKDIR /work
ENTRYPOINT ["wsinfer"]
CMD ["--help"]
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@stonybrookmedicine.edu>"
