#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
module load gcc/13.2.0
#module load cmake
export CC=gcc
export CXX=g++

# # Load Clang compiler
# module load clang/17.0.1
# export CC=clang
# export CXX=clang++

# Configure APEX
export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1
# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build
# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release \
	 -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	 $MKL_CONFIG
 # Build the project
make -j VERBOSE=1 all

################################################################################
# Run code
################################################################################
./gprat_cpp
