#!/bin/bash
set -x

# Utility functions for building the layers of docker images.

PREV_TAG=""

build_first() {
  TAG=$1
  DOCKERFILE=$2

  docker build \
    --file $DOCKERFILE \
    --tag $TAG .

  PREV_TAG=$TAG
}

build_next() {
  TAG=$1
  DOCKERFILE=$2

   docker build --progress=plain \
    --build-arg DOCKER_IMAGE_PARENT=$PREV_TAG \
    --file $DOCKERFILE \
    --tag $TAG .

  PREV_TAG=$TAG
}

export -f build_first
export -f build_next
