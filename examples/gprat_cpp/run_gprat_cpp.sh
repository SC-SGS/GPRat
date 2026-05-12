#!/bin/bash
# $1 cpu/cuda/sycl
# $2 mkl/none
# $3 SYCL with nvidia/amd/intel

################################################################################
set -e # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################

# Set device for computation
if [[ -z "$1" ]]; then
  echo "Input parameter is missing. Using default: Run computations on CPU"
elif [[ "$1" == "cuda" ]]; then
  use_gpu="--use_cuda"
elif [[ "$1" == "sycl" ]]; then
  use_gpu="--use_sycl"
elif [[ "$1" != "cpu" ]]; then
  echo "Please specify input parameter: cpu/cuda/sycl"
  exit 1
fi

if [[ "$3" == "nvidia" ]]; then

  CMAKE_PREFIX_PATH="/scratch-simcl1/grafml/Programs/oneMath_nvidia/oneMath/install/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

elif [[ "$3" == "amd" ]]; then

  CMAKE_PREFIX_PATH="/scratch-simcl1/grafml/Programs/oneMath_amd/oneMath/install/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

elif [[ "$3" == "intel" ]]; then

  echo "The Intel setup is not supported yet." 1>&2
  exit 1

fi

# Select BLAS library
if [[ "$2" == "mkl" ]]; then
  USE_MKL=ON
else
  USE_MKL=OFF
fi

# Set Spack if on simcl1n1, simcl1n2, simcl1n3, or simcl1n4
if [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" || "$HOSTNAME" == "simcl1n3" || "$HOSTNAME" == "simcl1n4" ]]; then

  spack_destination="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
  source $spack_destination/spack/share/spack/setup-env.sh

fi

if command -v spack &>/dev/null; then

  echo "Spack command found, checking for environments..."
  # Get current hostname
  HOSTNAME=$(hostname -s)

  if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then

    # Check if the gprat_cpu_gcc environment exists
    if spack env list | grep -q "gprat_cpu_gcc"; then
      echo "Found gprat_cpu_gcc environment, activating it."
      module load gcc/14.2.0
      export CXX=g++
      export CC=gcc
      spack env activate gprat_cpu_gcc
      GPRAT_WITH_CUDA=OFF # whether GPRAT_WITH_CUDA is ON of OFF is irrelevant for this example
      GPRAT_APEX_STEPS=OFF
      GPRAT_APEX_CHOLESKY=OFF
    fi

  elif [[ "$HOSTNAME" == "sven0" || "$HOSTNAME" == "sven1" ]]; then
    #module load gcc/13.2.1
    spack load openblas arch=linux-fedora38-riscv64
    HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
    GPRAT_WITH_CUDA=OFF
    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF
    ADD=64
  elif [[ $(uname -i) == "aarch64" ]]; then
    spack load gcc@14.2.0
    # Check if the gprat_cpu_arm environment exists
    if spack env list | grep -q "gprat_cpu_arm"; then
      echo "Found gprat_cpu_arm environment, activating it."
      spack env activate gprat_cpu_arm
    fi
    GPRAT_WITH_CUDA=OFF
    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF
    ADD=64

  elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF

    # Check if the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang

      if [[ "$1" == "cuda" ]]; then

        module load clang/17.0.1
        export CXX=clang++
        export CC=clang
        module load cuda/12.0.1
        GPRAT_WITH_CUDA=ON
        GPRAT_WITH_SYCL=OFF

      elif [[ "$1" == "sycl" ]]; then

        if command -v icpx --version &>/dev/null; then

          export CXX=icpx
          export CC=icx
          GPRAT_WITH_CUDA=OFF
          GPRAT_WITH_SYCL=ON

        else

          echo "DPC++ compiler not found. Please make sure that a DPC++ compiler is available in your PATH." 1>&2
          exit -1

        fi

      fi

    fi

  elif [[ "$HOSTNAME" == "simcl1n3" ]]; then
    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF

    # Check if the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang
      CMAKE_PREFIX_PATH="/scratch-simcl1/grafml/Programs/oneMath_nvidia/oneMath/install/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

      if [[ "$1" == "sycl" ]]; then

        if command -v icpx --version &>/dev/null; then

          export CXX=icpx
          export CC=icx
          GPRAT_WITH_CUDA=OFF
          GPRAT_WITH_SYCL=ON

        else

          echo "DPC++ compiler not found. Please make sure that a DPC++ compiler is available in your PATH." 1>&2
          exit -1

        fi

      fi

    fi

  elif [[ "$HOSTNAME" == "pcsgs04" ]]; then

    echo "Host pcsgs04 is currently not supported." 1>&2
    exit -1

  else

    echo "Hostname is $HOSTNAME — no action taken."
  fi

else

  echo "Spack command not found. Building example without Spack."
  # Assuming that Spack is not required on given system
fi

# Configure APEX
export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

################################################################################
# Compile code
################################################################################

rm -rf build && mkdir build && cd build && mkdir run_gprat_cpp && cd run_gprat_cpp

# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGPRat_DIR=./lib$ADD/cmake/GPRat \
  -DGPRAT_WITH_CUDA=${GPRAT_WITH_CUDA} \
  -DGPRAT_WITH_SYCL=${GPRAT_WITH_SYCL} \
  -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
  -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY} \
  -DHPX_DIR=$HPX_CMAKE \
  -DUSE_MKL=$USE_MKL \
  -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH

# Build the project
make -j

################################################################################
# Run code
################################################################################
echo "Running GPRat C++ example"
./gprat_cpp $use_gpu
echo "Finished running GPRat C++ example"
