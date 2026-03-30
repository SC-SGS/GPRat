#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

###################################################################################################
# Parameters
###################################################################################################
# $1: toggle gpflow             yes/no
# $2: toggle gpytorch           yes/no
# $3: toggle GPRat Cholesky     yes/no
# $4: toggle GPRat Python       yes/no
# $5: hardware                  cpu/gpu
# $6: vendor                    nvidia/amd/cpu
# $7: toggle GPRat CUDA/SYCL    cuda/sycl           only for NVIDIA GPUs

# Set global variables
HARDWARE=$5
VENDOR=$6

# Set variables relevant for GPRat
if [[ "$HARDWARE" == "cpu" ]]; then

    GPRAT_TARGET="cpu"
    GPRAT_USE_GPU=""
    BUILD_DIR="build/release-linux"

elif [[ "$HARDWARE" == "gpu" ]]; then

    GPRAT_USE_GPU="--use-gpu"

    if [[ "$VENDOR" == "nvidia" ]]; then

        if [[ "$7" == "cuda" ]]; then

            GPRAT_TARGET="cuda"
            BUILD_DIR="build/release-linux-cuda"

        elif [[ "$7" == "sycl" ]]; then

            GPRAT_TARGET="sycl"
            BUILD_DIR="build/release-linux-sycl"

        else

            echo -e "\e[31mUnsupported GPRat target: ${5}\e[0m"
            exit 1

        fi

    elif [[ "$VENDOR" == "amd" ]]; then

        GPRAT_TARGET="sycl"
        BUILD_DIR="build/release-linux-sycl"

    else

        echo -e "\e[31mUnsupported GPU vendor: ${VENDOR}\e[0m"
        exit 1

    fi

else

    echo -e "\e[31mUnsupported hardware type: ${HARDWARE}\e[0m"
    exit 1

fi

# Create benchmark folder
mkdir -p benchmark_results_${HARDWARE}_${VENDOR}

# GPflow
if [[ "$1" == "yes" ]]; then
    echo -e "\e[32mRunning GPflow benchmarks on ${HARDWARE} ${VENDOR}...\e[0m"

    
    cd examples/gpflow_reference
    ./run_gpflow.sh ${HARDWARE} ${VENDOR} > /dev/null
    cp output.csv ../../benchmark_results_${HARDWARE}_${VENDOR}/gpflow_${VENDOR}.csv
    rm -rf gpflow_${HARDWARE}_env
    cd ../..

else

    echo -e "\e[33mSkipping GPflow benchmarks.\e[0m"

fi

# GPyTorch
if [[ "$2" == "yes" ]]; then
    echo -e "\e[32mRunning GPyTorch benchmarks on ${HARDWARE} ${VENDOR}...\e[0m"

    cd examples/gpytorch_reference
    ./run_gpytorch.sh ${HARDWARE} ${VENDOR} > /dev/null
    cp output.csv ../../benchmark_results_${HARDWARE}_${VENDOR}/gpytorch_${VENDOR}.csv
    rm -rf gpytorch_${HARDWARE}_env
    cd ../..

else   

    echo -e "\e[33mSkipping GPyTorch benchmarks.\e[0m"

fi

# GPRat compile
if [[ "$3" == "yes" || $4 == "yes" ]]; then
    
    echo -e "\e[32mCompiling GPRat for ${VENDOR} ${HARDWARE} with ${GPRAT_TARGET} target...\e[0m"

    rm -rf build

    ./compile_gprat.sh cpp $GPRAT_TARGET release mkl > /dev/null
    ./compile_gprat.sh python $GPRAT_TARGET release mkl > /dev/null

else

    echo -e "\e[33mNot compiling GPRat.\e[0m"

fi

# GPRat Cholesky
if [[ "$3" == "yes" ]]; then
    echo -e "\e[32mRunning GPRat Cholesky benchmarks on ${VENDOR} ${HARDWARE} with ${GPRAT_TARGET} target...\e[0m"

    cd examples/gprat_cpp
    end_cores=$(python3 -c "import json; print(json.load(open('config.json'))['END_CORES'])")
    core_count=$((end_cores * 2))
    cd ../..

    cd ${BUILD_DIR}/examples/gprat_cpp/
    taskset -c 0-$core_count:2 ./gprat_cpp $GPRAT_USE_GPU > /dev/null
    cp ../output.csv ../../../../benchmark_results_${HARDWARE}_${VENDOR}/gprat_cholesky_${VENDOR}_${GPRAT_TARGET}.csv
    cd ../../../..

else

    echo -e "\e[33mSkipping GPRat Cholesky benchmarks.\e[0m"

fi

# GPRat Python
if [[ "$4" == "yes" ]]; then
    echo -e "\e[32mRunning GPRat Python benchmarks on ${VENDOR} ${HARDWARE} with ${GPRAT_TARGET} target...\e[0m"

    cd examples/gprat_python
    ./run_gprat_python.sh ${GPRAT_TARGET} ${VENDOR} > /dev/null
    cp output.csv ../../benchmark_results_${HARDWARE}_${VENDOR}/gprat_python_${VENDOR}_${GPRAT_TARGET}.csv
    cd ../..

else

    echo -e "\e[33mSkipping GPRat Python benchmarks.\e[0m"

fi

echo -e "\e[32mCopying results to home directory... \e[0m"
mkdir -p ${HOME}/GPRAT-BENCHMARKS

cp -r benchmark_results_${HARDWARE}_${VENDOR}/ ${HOME}/GPRAT-BENCHMARKS/

echo -e "\e[32mDone.\e[0m"
