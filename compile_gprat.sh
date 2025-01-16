#!/bin/bash
# $1: python/cpp
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
module load gcc/13.2.0
export CC=gcc
export CXX=g++
# Activate spack environment
spack env activate gprat_cpu_gcc

# # Load Clang compiler
# module load clang/17.0.1
# export CC=clang
# export CXX=clang++
# # Activate spack environment
# spack env activate gprat_gpu_clang

# Configure APEX
#export APEX_SCREEN_OUTPUT=1

# Bindings
if [[ "$1" == "python" ]]
then
	export BINDINGS=ON
	export INSTALL_DIR=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]
then
	export BINDINGS=OFF
	export INSTALL_DIR=$(pwd)/examples/gprat_cpp
else
    echo "Please specify input parameter: python/cpp"
    exit 1
fi

# Release:	release-linux
# Debug:	dev-linux
export PRESET=release-linux

################################################################################
# Compile code
################################################################################
cmake --preset $PRESET \
      -DGPRAT_BUILD_BINDINGS=$BINDINGS \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
      -DGPRAT_ENABLE_FORMAT_TARGETS=OFF
cmake --build --preset $PRESET
cmake --install build/$PRESET
# ctest --preset $PRESET
