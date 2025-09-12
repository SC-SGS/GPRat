#!/bin/bash
# $1 cpu/gpu

################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################

if command -v spack &> /dev/null; then
    echo "Spack command found."
    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	module load gcc/14.2.0
	export CXX=g++
	export CC=gcc
	spack install gprat%gcc@14.2.0 blas=openblas
	spack load gprat blas=openblas
    elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
	# Check if the gprat_gpu_clang environment exists
	    module load clang/17.0.1
	    export CXX=clang++
	    export CC=clang
	    spack install gprat%clang@17.0.1 blas=openblas +cuda
	    spack load gprat blas=openblas +cuda
    else
    	echo "Hostname is $HOSTNAME — no action taken."
    fi
else
    echo "Spack command not found."
    exit 1
fi

# Configure APEX
export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build

# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make -j

################################################################################
# Run code
################################################################################
if [[ -z "$1" ]]; then
    echo "Input parameter is missing. Using default: Run computations on CPU"
elif [[ "$1" == "gpu" ]]; then
    use_gpu="--use_gpu"
elif [[ "$1" != "cpu" ]]; then
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi

./gprat_cpp $1
