#!/bin/bash
set -ev
cd /build
mkdir build-native
cd build-native
cmake ..
make -j 2 all
cd /build
mkdir build-cross
cd build-cross
cmake -DCMAKE_TOOLCHAIN_FILE=../travis/toolchain-aarch64.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DEMULATOR=qemu-aarch64-static -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
