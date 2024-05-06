#!/bin/bash
set -ex

# Build the local docker image, see the Taskfile.yml for usage.

if [ $(uname -m) == 'arm64' ]; then
  # Containers are usually multi-architecture, and so an AMD machine such as the new
  # M1 processers for macbook will choose AMD over x86. The translations C++ code doesn't
  # usually work nicely with AMD, so the Docker containers need to be forced to use x86_64
  # architectures. This is quite slow, but will still be a usable Docker container.
  echo "Overriding the arm64 architecture as amd64.";
  export DOCKER_DEFAULT_PLATFORM=linux/amd64;
fi

source ./utils/tasks/docker-build.sh

# Base the images off of the taskcluster images.
build_first ftt-taskcluster-base-amd64  taskcluster/docker/base/Dockerfile
build_next  ftt-taskcluster-test-amd64  taskcluster/docker/test/Dockerfile

# Build out the local version.
build_next  ftt-local-train-amd64  docker/local-train.Dockerfile
build_next  ftt-local-test-amd64   docker/local-test.Dockerfile
