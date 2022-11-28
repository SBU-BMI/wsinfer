#!/usr/bin/env bash
#
# Build all Docker images. This includes the main image and all app images.

set -ex

here=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $here/..

usage="usage: $0 VERSION [PUSH_IMAGES]"

if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit 1
fi

if [ -z "$1" ]; then
    echo "No version supplied"
    echo $usage
    exit 2
fi

# Version of the pipeline.
version=$1

# Main image.
docker build -t kaczmarj/patch-classification-pipeline:$version .

# TILs
docker build -t kaczmarj/patch-classification-pipeline:$version-tils - < dockerfiles/tils.dockerfile

# Tumor BRCA
docker build -t kaczmarj/patch-classification-pipeline:$version-tumor-brca - < dockerfiles/tumor-brca.dockerfile

# Tumor LUAD
docker build -t kaczmarj/patch-classification-pipeline:$version-tumor-luad - < dockerfiles/tumor-luad.dockerfile

# Tumor PAAD
docker build -t kaczmarj/patch-classification-pipeline:$version-tumor-paad - < dockerfiles/tumor-paad.dockerfile

# Tumor PRAD
docker build -t kaczmarj/patch-classification-pipeline:$version-tumor-prad - < dockerfiles/tumor-prad.dockerfile

# Push images.
push_images="${2:-0}"
if [ $push_images -eq 0 ]; then
    echo "Not pushing images. Pass a 1 to the script to push images."
else
    echo "Pushing images."
    docker push kaczmarj/patch-classification-pipeline:$version
    docker push kaczmarj/patch-classification-pipeline:$version-tils
    docker push kaczmarj/patch-classification-pipeline:$version-tumor-brca
    docker push kaczmarj/patch-classification-pipeline:$version-tumor-luad
    docker push kaczmarj/patch-classification-pipeline:$version-tumor-paad
    docker push kaczmarj/patch-classification-pipeline:$version-tumor-prad
fi
