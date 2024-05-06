# A Docker image to run tests or experiment locally on a CPU machine
# It should be based on /taskcluster/docker/test/Dockerfile

ARG DOCKER_IMAGE_PARENT
FROM $DOCKER_IMAGE_PARENT

# Install the tools necessary to build the training binaries.
# This should be similar to taskcluster/docker/toolchain-build/Dockerfile
RUN apt-get update -qq \
    && apt-get install -y wget \
                          zip \
                          build-essential \
                          gcc \
                          g++ \
                          make \
                          cmake \
                          libboost-dev \
                          libboost-all-dev \
                          tar \
                          libxml2 \
                          libhunspell-dev \
                          bc \
                          libopenblas-dev \
                          openssl \
                          libssl-dev  \
    && apt-get clean

# The binary tools are all placed in `/builds/worker/tools/bin`
RUN mkdir -p /builds/worker/tools/bin && \
    chown -R worker:worker /builds/worker/tools

# The pipeline/setup scripts place the binaries in this folder.
ENV BIN=/builds/worker/tools/bin

WORKDIR /builds/worker/tools

# The setup tools contain all of the scripts to build the training binaries.
COPY pipeline/setup setup

RUN git clone https://github.com/marian-nmt/extract-lex.git extract-lex
RUN ./setup/compile-extract-lex.sh extract-lex/build $(nproc)

RUN git clone https://github.com/clab/fast_align.git fast_align
RUN ./setup/compile-fast-align.sh fast_align/build $(nproc)

RUN git clone https://github.com/kpu/preprocess.git preprocess
RUN ./setup/compile-preprocess.sh preprocess/build $(nproc)

# Setup Marian.
# Use the same revision as in taskcluster/kinds/fetch/toolchains.yml
RUN git clone https://github.com/marian-nmt/marian-dev.git marian-dev
RUN cd marian-dev && git checkout 2d067afb9ce5e3a0b6c32585706affc6e7295920
RUN ./setup/compile-marian.sh marian-dev/build $(nproc) false
RUN mv /builds/worker/tools/marian-dev/build $BIN/marian-dev
ENV MARIAN=$BIN/marian-dev

# Clean up the checkouts.
RUN rm -rf \
  setup \
  extract-lex \
  fast_align \
  preprocess \
  marian-dev \
