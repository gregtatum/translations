#!/bin/bash
set -x

# Run the local docker image, see the Taskfile.yml for usage.

echo 'Running docker run'

source utils/tasks/docker-setup.sh

if [ -d .venv ] || [ -d venv ];
then
    set +x
    echo "Virtual environments located in the project directory are not supported."
    echo "Change your poetry configuration, and remove the .venv or venv directories:"
    echo "Run:"
    echo "> rm -rf venv .venv"
    echo "> poetry config virtualenvs.in-project false"
    exit 1
fi


echo 'Running docker run'

docker run \
  --interactive \
  --tty \
  --rm \
  --volume $(pwd):/builds/worker/checkouts \
  --workdir /builds/worker/checkouts \
  ftt-local "$@"
