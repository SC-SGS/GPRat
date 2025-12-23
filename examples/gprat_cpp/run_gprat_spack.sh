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
	spack load gprat%gcc blas=openblas
    elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
	# Check if the gprat_gpu_clang environment exists
	    module load clang/17.0.1
	    export CXX=clang++
	    export CC=clang
	    GPRAT_WITH_CUDA=ON
	    GPRAT_APEX_STEPS=OFF
	    GPRAT_APEX_CHOLESKY=OFF
	    spack install gprat%clang@17.0.1 blas=openblas +cuda cuda_arch=80 ^cmake@3.30.5
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
cmake .. -DCMAKE_BUILD_TYPE=Release \
	 -DGPRAT_WITH_CUDA=${GPRAT_WITH_CUDA} \
	 -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
	 -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY}

# Build the project
make -j

################################################################################
# Run code
################################################################################
if [[ -z "$1" ]]; then
    echo "Input parameter is missing. Using default: Run computations on CPU"
elif [[ "$1" == "gpu" ]]; then
    GPU="--use_gpu"
elif [[ "$1" != "cpu" ]]; then
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi

./gprat_cpp $GPU
