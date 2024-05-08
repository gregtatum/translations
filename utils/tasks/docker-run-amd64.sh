#!/bin/bash
set -x

# Run the local docker image, see the Taskfile.yml for usage.

export DOCKER_DEFAULT_PLATFORM=linux/amd64;

echo 'Running docker run'

docker run \
  --interactive \
  --tty \
  --rm \
  --volume $(pwd):/builds/worker/checkouts \
  --workdir /builds/worker/checkouts \
  ftt-local-test-amd64 "$@"
