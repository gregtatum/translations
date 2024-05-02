#!/bin/bash
set -x

# Build the local docker image, see the Taskfile.yml for usage. This uses the machine's
# underlying chip architecture, which is not guaranteed to build certain training binaries.

docker build \
  --file taskcluster/docker/base/Dockerfile \
  --tag ftt-base-native .

docker build \
  --build-arg DOCKER_IMAGE_PARENT=ftt-base-native \
  --file taskcluster/docker/test/Dockerfile \
  --tag ftt-test-native .

docker build \
  --build-arg DOCKER_IMAGE_PARENT=ftt-test-native \
  --file docker/local/Dockerfile \
  --tag ftt-local-native .
