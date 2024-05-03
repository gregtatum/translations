# A Docker image to run tests or experiment locally on a CPU machine
# It should be based on /taskcluster/docker/test/Dockerfile

ARG DOCKER_IMAGE_PARENT
FROM $DOCKER_IMAGE_PARENT

WORKDIR /builds/worker/tools

RUN git clone https://github.com/marian-nmt/extract-lex.git extract-lex
RUN ./compile-extract-lex.sh extract-lex/build $(nproc)

RUN git clone https://github.com/clab/fast_align.git fast_align
RUN ./compile-fast-align.sh fast_align/build $(nproc)

RUN git clone https://github.com/kpu/preprocess.git preprocess
RUN ./compile-preprocess.sh preprocess/build $(nproc)

RUN git clone https://github.com/marian-nmt/marian-dev.git marian-dev
# Use the same revision as in taskcluster/kinds/fetch/toolchains.yml
# it corresponds to v1.12.14 2d067afb 2024-02-16 11:44:13 -0500
RUN cd marian-dev && git checkout 2d067afb9ce5e3a0b6c32585706affc6e7295920
RUN ./compile-marian.sh marian-dev/build $(nproc) false

ENV MARIAN=/builds/worker/tools/marian-dev/build
