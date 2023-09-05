# FROM $DOCKER_IMAGE_PARENT
FROM base:1.0
LABEL maintainer="Mozilla Release Engineering <release+docker@mozilla.com>"

RUN pip install poetry
RUN poetry config virtualenvs.create false
# RUN poetry install

VOLUME /builds/worker/checkouts
