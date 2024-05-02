#!/bin/bash
set -x

# Run the local docker image, see the Taskfile.yml for usage.

if [ $(uname -m) == 'arm64' ]; then
  echo "Overriding the arm64 architecture as amd64.";
  export DOCKER_DEFAULT_PLATFORM=linux/amd64;
fi

echo 'Running docker run'

docker run \
  --interactive \
  --tty \
  --rm \
  --volume $(pwd):/builds/worker/checkouts \
  --workdir /builds/worker/checkouts \
  ftt-local-train-amd64 "$@"
