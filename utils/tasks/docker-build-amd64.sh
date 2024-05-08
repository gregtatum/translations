#!/bin/bash
set -ex

# Build the local docker image, see the Taskfile.yml for usage.

# The base image is multi-platform, but the training binaries require amd64 CPUs.
# In some systems (such as macOS) the CPU instructions will be emulated.
export DOCKER_DEFAULT_PLATFORM=linux/amd64;

source ./utils/tasks/docker-build.sh

# Base the images off of the taskcluster images.
build_first ftt-taskcluster-base-amd64  taskcluster/docker/base/Dockerfile
build_next  ftt-taskcluster-test-amd64  taskcluster/docker/test/Dockerfile

# Build out the local version.
build_next  ftt-local-train-amd64  docker/local-train.Dockerfile
build_next  ftt-local-test-amd64   docker/local-test.Dockerfile
