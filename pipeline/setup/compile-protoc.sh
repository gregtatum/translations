#!/bin/bash
set -ex

##
# Usage: ./compile-protoc.sh ./protobuf
#

test -v BIN

# This follows the instructions from: https://github.com/protocolbuffers/protobuf/blob/v3.14.0/src/README.md
cd $1
git submodule update --init --recursive
./autogen.sh

./configure
make
make check
make install
ldconfig # refresh shared library cache.

# Now that we're done, copy the bin.
cp ./src/protoc "${BIN}"

# Link in the includes.
ln -s $(pwd)/src/google /usr/include/google
