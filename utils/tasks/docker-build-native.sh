#!/bin/bash
set -ex

# Build the local docker image, see the Taskfile.yml for usage. This uses the machine's
# underlying chip architecture, which is not guaranteed to build certain training binaries.

source ./utils/tasks/docker-build.sh

# Base the images off of the taskcluster images.
build_first ftt-taskcluster-base-native  taskcluster/docker/base/Dockerfile
build_next  ftt-taskcluster-test-native  taskcluster/docker/test/Dockerfile

# Build out the local version.
build_next  ftt-local-test-native  docker/local-test.Dockerfile
