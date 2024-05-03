#!/bin/bash
set -x

# Build the local docker image, see the Taskfile.yml for usage.

# Containers are usually multi-architecture, and so an AMD machine such as the new
# M1 processers for macbook will choose AMD over x86. The translations C++ code doesn't
# usually work nicely with AMD, so the Docker containers need to be forced to use x86_64
# architectures. This is quite slow, but will still be a usable Docker container.
if [ $(uname -m) == 'arm64' ]; then
  echo "Overriding the arm64 architecture as amd64.";
  export DOCKER_DEFAULT_PLATFORM=linux/amd64;
fi

docker build \
  --file taskcluster/docker/base/Dockerfile \
  --tag ftt-base-amd64 .

docker build \
  --build-arg DOCKER_IMAGE_PARENT=ftt-base-amd64 \
  --file taskcluster/docker/test/Dockerfile \
  --tag ftt-test-amd64 .

docker build \
  --build-arg DOCKER_IMAGE_PARENT=ftt-test-amd64 \
  --file docker/local-test.Dockerfile \
  --tag ftt-local-amd64 .

docker build \
  --build-arg DOCKER_IMAGE_PARENT=ftt-local-amd64 \
  --file docker/local-train.Dockerfile \
  --tag ftt-local-train-amd64 .
