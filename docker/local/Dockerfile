# A Docker image to run tests or experiment locally on a CPU machine
# It should be based on /taskcluster/docker/test/Dockerfile

ARG DOCKER_IMAGE_PARENT
FROM $DOCKER_IMAGE_PARENT

# Similar to /taskcluster/docker/toolchain/Dockerfile
RUN apt-get update -qq \
    # We need to install tzdata before all of the other packages. Otherwise it will show an interactive dialog that
    # we cannot navigate while building the Docker image.
    && apt-get install -y tzdata \
    && apt-get install -y wget \
                          curl \
                          zip \
                          build-essential \
                          gcc \
                          g++ \
                          make \
                          cmake \
                          libboost-dev \
                          libboost-all-dev \
                          zstd \
                          tar \
                          libxml2 \
                          libhunspell-dev \
                          bc \
                          libopenblas-dev \
                          openssl \
                          libssl-dev  \
                          python3.10-venv \
         && apt-get clean

RUN mkdir /builds/worker/tools && \
    chown worker:worker /builds/worker/tools && \
    mkdir /builds/worker/tools/bin && \
    chown worker:worker /builds/worker/tools/bin

WORKDIR /builds/worker/tools

ADD pipeline/setup/* .

ENV BIN=/builds/worker/tools/bin

# Work around poetry not finding `python`.
# https://github.com/python-poetry/poetry/issues/6371
RUN ln -s /bin/python3 /bin/python

# Have poetry ignore any venvs in the workdir by using a system install.
RUN poetry config virtualenvs.create false

# Install taskfile - https://taskfile.dev/
# Keep the version in sync with taskcluster/docker/test/Dockerfile.
RUN curl -sSLf "https://github.com/go-task/task/releases/download/v3.35.1/task_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin

# In some operating systems, the .git configuration will complain if the users
# do not match for the .git. Since this is a local environment, we don't worry about
# permissions escalation exploits that may be an issue in a production docker container.
RUN git config --global --add safe.directory /builds/worker/checkouts

# Allow scripts to detect if they are running in docker
ENV IS_DOCKER 1

# https://stackoverflow.com/questions/39539110/pyvenv-not-working-because-ensurepip-is-not-available
ENV LC_ALL="en_US.UTF-8"
ENV LC_CTYPE="en_US.UTF-8"
