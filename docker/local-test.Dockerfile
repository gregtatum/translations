# A Docker image to run tests or experiment locally on a CPU machine
# It should be based on /taskcluster/docker/test/Dockerfile

ARG DOCKER_IMAGE_PARENT
FROM $DOCKER_IMAGE_PARENT

# Work around poetry not finding `python`.
# https://github.com/python-poetry/poetry/issues/6371
RUN ln -s /bin/python3 /bin/python

# Install taskfile - https://taskfile.dev/
# Keep the version in sync with taskcluster/docker/test/Dockerfile.
RUN curl -sSLf "https://github.com/go-task/task/releases/download/v3.35.1/task_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin

# In some operating systems, the .git configuration will complain if the users
# do not match for the .git. Since this is a local environment, we don't worry about
# permissions escalation exploits that may be an issue in a production docker container.
RUN git config --global --add safe.directory /builds/worker/checkouts
