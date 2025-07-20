#!/bin/bash
set -x

# Build the local docker image, see the Taskfile.yml for usage.

source utils/tasks/docker-setup.sh

VARIANT="$1"
if [[ "$VARIANT" != "test" && "$VARIANT" != "inference" ]]; then
  echo "Error: First argument must be 'test' or 'inference'"
  exit 1
fi

docker build \
  --file "taskcluster/docker/base/Dockerfile" \
  --tag translations-tc-base \
  taskcluster/docker/base # build context

docker build \
  --build-arg DOCKER_IMAGE_PARENT=translations-base \
  --file "$TC_DOCKER_TEST_PATH/Dockerfile" \
  --tag translations-tc-test \
  taskcluster/docker/test # build context

# Build the local images
docker build \
  --build-arg DOCKER_IMAGE_PARENT=translations-local \
  --file "docker/Dockerfile.local" \
  --tag translations-local \
  . # build context

# Build either translations-local-test or translations-local-inference
docker build \
  --build-arg DOCKER_IMAGE_PARENT=translations-local-$VARIANT \
  --file "docker/Dockerfile.$VARIANT" \
  --tag translations-local-$VARIANT \
  . # build context
