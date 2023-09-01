FROM $DOCKER_IMAGE_PARENT
LABEL maintainer="Mozilla Release Engineering <release+docker@mozilla.com>"

# TODO - Where is app?
COPY /app /app
# TODO - Why is this here?
COPY pyproject.toml /app

WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry

# There is no reason to use a virtualenv, as the Docker runtime is already virtualized.
RUN poetry config virtualenvs.create false
RUN poetry install
