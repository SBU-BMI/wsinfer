#!/usr/bin/env bash
#
# Build all Docker images. This includes the main image and all app images.

set -e

here=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $here/..

usage="usage: $0 VERSION [PUSH_IMAGES]"

if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    echo $usage
    exit 1
fi

if [ -z "$1" ]; then
    echo "No version supplied"
    echo $usage
    exit 2
fi

# Version of the pipeline.
version=$1

BASE_IMAGE="kaczmarj/wsinfer"

build () {
    tag=$1
    file=$2
    name=$BASE_IMAGE:$tag
    echo "Building $name from $file"
    if ! grep --quiet "FROM kaczmarj/wsinfer:$version" $file; then
        echo "Base image doesn't look right..."
        exit 1
    fi
    docker build -t $name - < $file
}

# Main image.
docker build -t kaczmarj/wsinfer:$version .

# TILs
build $version-tils dockerfiles/tils.dockerfile

# TILs VGG16
build $version-tils-vgg16 dockerfiles/tils-vgg16.dockerfile

# Tumor BRCA
build $version-tumor-brca dockerfiles/tumor-brca.dockerfile

# Tumor LUAD
build $version-tumor-luad dockerfiles/tumor-luad.dockerfile

# Tumor PAAD
build $version-tumor-paad dockerfiles/tumor-paad.dockerfile

# Tumor PRAD
build $version-tumor-prad dockerfiles/tumor-prad.dockerfile

# Push images.
push_images="${2:-0}"
if [ $push_images -eq 0 ]; then
    echo "Not pushing images. Pass a 1 to the script to push images."
else
    echo "Pushing images."
    docker push kaczmarj/wsinfer:$version
    docker push kaczmarj/wsinfer:$version-tils
    docker push kaczmarj/wsinfer:$version-tils-vgg16
    docker push kaczmarj/wsinfer:$version-tumor-brca
    docker push kaczmarj/wsinfer:$version-tumor-luad
    docker push kaczmarj/wsinfer:$version-tumor-paad
    docker push kaczmarj/wsinfer:$version-tumor-prad
fi
