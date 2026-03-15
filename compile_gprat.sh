#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

################################################################################
# Parameters
################################################################################
# $1: python/cpp
# $2: cpu/cuda/sycl
# $3: release/dev
# $4: mkl

################################################################################
# Bindings
################################################################################
if [[ "$1" == "python" ]]; then
  BINDINGS=ON
  INSTALL_DIR=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]; then
  BINDINGS=OFF
  INSTALL_DIR=$(pwd)/examples/gprat_cpp
else
  echo "Please specify input parameter: python/cpp"
  exit 1
fi

################################################################################
# CMake Presets
################################################################################
if [[ "$2" == "cpu" ]]; then
  if [[ "$3" == "release" ]]; then
    PRESET=release-linux
  elif [[ "$3" == "dev" ]]; then
    PRESET=dev-linux
  else
    echo "Input parameter for release or dev mode is missing. Using default: Build in Release mode"
    PRESET=release-linux
  fi
elif [[ "$2" == "cuda" ]]; then
  if [[ "$3" == "release" ]]; then
    PRESET=release-linux-cuda
  elif [[ "$3" == "dev" ]]; then
    PRESET=dev-linux-cuda
  else
    echo "Input parameter for release or dev mode is missing. Using default: Build in Release mode"
    PRESET=release-linux-cuda
  fi
elif [[ "$2" == "sycl" ]]; then
  if [[ "$3" == "release" ]]; then
    PRESET=release-linux-sycl
  elif [[ "$3" == "dev" ]]; then
    PRESET=dev-linux-sycl
  else
    echo "Input parameter for release or dev mode is missing. Using default: Build in Release mode"
    PRESET=release-linux-sycl
  fi
elif [[ "$2" != "cpu" ]]; then
  echo "Input parameter is not any of {cpu,cuda,sycl}. Using default: Run computations on CPU in Release mode"
  PRESET=release-linux
fi

################################################################################
# Select BLAS library
################################################################################
if [[ "$4" == "mkl" ]]; then
  USE_MKL=ON
else
  USE_MKL=OFF
fi

# Select APEX profiling option
if [[ "$4" == "steps" ]]; then
  GPRAT_APEX_STEPS=ON
  GPRAT_APEX_CHOLESKY=OFF
elif [[ "$4" == "cholesky" ]]; then
  GPRAT_APEX_STEPS=OFF
  GPRAT_APEX_CHOLESKY=ON
else
  GPRAT_APEX_STEPS=OFF
  GPRAT_APEX_CHOLESKY=OFF
fi

################################################################################
# Developer: Pick Spack installation depending on the host
################################################################################

# Set Spack if on simcl1n1, simcl1n2, simcl1n3, or simcl1n4
if [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" || "$HOSTNAME" == "simcl1n3" || "$HOSTNAME" == "simcl1n4" ]]; then

  spack_destination="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
  source $spack_destination/spack/share/spack/setup-env.sh

fi

# Set Spack if on psgs04
if [[ "$HOSTNAME" == "pcsgs04" ]]; then

  spack_destination="/scratch/grafml/gprat-spack/spack/"
  source $spack_destination/share/spack/setup-env.sh

fi

################################################################################
# Setup Compilation Requirements
################################################################################

# Assuming Spack is found
if command -v spack &>/dev/null; then
  echo "Spack command found, checking for environments..."

  # Get current hostname
  HOSTNAME=$(hostname -s)

  # ipvs-epyc1
  if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then

    # Check whether the gprat_cpu_gcc environment exists
    if spack env list | grep -q "gprat_cpu_gcc"; then
      echo "Found gprat_cpu_gcc environment, activating it."
      module load gcc/14.2.0
      export CXX=g++
      export CC=gcc
      spack env activate gprat_cpu_gcc
      GPRAT_WITH_CUDA=OFF # whether GPRAT_WITH_CUDA is ON of OFF is irrelevant for this example
    fi

    # sven0 and sven1
  elif [[ "$HOSTNAME" == "sven0" || "$HOSTNAME" == "sven1" ]]; then

    #module load gcc/13.2.1
    spack load openblas arch=linux-fedora38-riscv64
    HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
    USE_MKL=OFF

    # aarch64
  elif [[ $(uname -i) == "aarch64" ]]; then

    spack load gcc@14.2.0
    # Check if the gprat_cpu_arm environment exists
    if spack env list | grep -q "gprat_cpu_arm"; then
      echo "Found gprat_cpu_arm environment, activating it."
      spack env activate gprat_cpu_arm
    fi
    USE_MKL=OFF

    # simcl1n1 and simcl1n2 with NVIDIA GPUs
  elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then

    # Check if the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang

      if [[ "$2" == "cuda" ]]; then

        module load clang/17.0.1
        export CXX=clang++
        export CC=clang
        module load cuda/12.0.1
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')
        GPRAT_WITH_CUDA=ON
        GPRAT_WITH_SYCL=OFF

      elif [[ "$2" == "sycl" ]]; then

        if command -v icpx --version &>/dev/null; then

          CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')
          export CXX=icpx
          export CC=icx
          GPRAT_WITH_CUDA=OFF
          GPRAT_WITH_SYCL=ON
          GPRAT_SYCL_NVIDIA=ON
          CMAKE_PREFIX_PATH="/scratch-simcl1/grafml/Programs/oneMath_nvidia/oneMath/install/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

        else

          echo "DPC++ compiler not found. Please make sure that a DPC++ compiler is available in your PATH." 1>&2
          exit -1

        fi

      fi

    fi

  # simcl1n3 with AMD GPUs
  elif [[ "$HOSTNAME" == "simcl1n3" ]]; then
    echo "Support for host simcl1n3 is currently experimental."

    # Check if the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang

      if [[ "$2" == "sycl" ]]; then

        if command -v icpx --version &>/dev/null; then
          export CXX=icpx
          export CC=icx
          GPRAT_WITH_CUDA=OFF
          GPRAT_WITH_SYCL=ON
          GPRAT_SYCL_AMD=ON
          HIP_TARGETS="gfx90a"
          CMAKE_PREFIX_PATH="/scratch-simcl1/grafml/Programs/oneMath_amd/oneMath/install/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

        else

          echo "DPC++ compiler not found. Please make sure that a DPC++ compiler is available in your PATH." 1>&2
          exit -1

        fi
      fi
    fi

  # simcl1n4 without GPU
  elif [[ "$HOSTNAME" == "simcl1n4" ]]; then

    # Check if the gprat_cpu_gcc environment exists
    if spack env list | grep -q "gprat_cpu_gcc"; then

      echo "Found gprat_cpu_gcc environment, activating it."
      module load gcc/14.2.0
      export CXX=g++
      export CC=gcc
      spack env activate gprat_cpu_gcc

    fi

  # pcsgs04 with Intel GPU
  elif [[ "$HOSTNAME" == "pcsgs04" ]]; then
    # Check if the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang

      if [[ "$2" == "sycl" ]]; then

        if command -v icpx --version &>/dev/null; then

          export CXX=icpx
          export CC=icx
          GPRAT_WITH_CUDA=OFF
          GPRAT_WITH_SYCL=ON
          GPRAT_SYCL_INTEL=ON
          CMAKE_PREFIX_PATH="/scratch/grafml/oneMath_intel_v0.7/oneMath/install:${CMAKE_PREFIX_PATH:-}"

        else

          echo "DPC++ compiler not found. Please make sure that a DPC++ compiler is available in your PATH." 1>&2
          exit -1

        fi
      
      fi

    fi

    # invalid hostnames
  else

    echo "Hostname is $HOSTNAME — no action taken."

  fi

# Assuming Spack is not found
else

  echo "Spack command not found. Building example without Spack."

fi

################################################################################
# Set up CMake
################################################################################

# CPU build
if [[ $PRESET == "release-linux" || $PRESET == "dev-linux" ]]; then

  cmake --preset $PRESET \
    -DGPRAT_BUILD_BINDINGS=$BINDINGS \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
    -DHPX_DIR=$HPX_CMAKE \
    -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
    -DGPRAT_ENABLE_MKL=$USE_MKL \
    -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
    -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY}

# CUDA build
elif [[ $PRESET == "release-linux-cuda" || $PRESET == "dev-linux-cuda" ]]; then
  CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')

  cmake --preset $PRESET \
    -DGPRAT_BUILD_BINDINGS=$BINDINGS \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
    -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
    -DGPRAT_ENABLE_MKL=$USE_MKL \
    -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
    -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY} \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_COMPILER=$(which clang++) \
    -DCMAKE_CUDA_FLAGS=--cuda-path=${CUDA_HOME} \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}

# SYCL build
elif [[ $PRESET == "release-linux-sycl" || $PRESET == "dev-linux-sycl" ]]; then

  cmake --preset $PRESET \
    -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
    -DGPRAT_BUILD_BINDINGS=$BINDINGS \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
    -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
    -DGPRAT_ENABLE_MKL=$USE_MKL \
    -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
    -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY} \
    -DCMAKE_C_COMPILER=$(which icx) \
    -DCMAKE_CXX_COMPILER=$(which icpx) \
    -DGPRAT_WITH_SYCL=ON \
    -DGPRAT_SYCL_NVIDIA=$GPRAT_SYCL_NVIDIA \
    -DGPRAT_SYCL_AMD=$GPRAT_SYCL_AMD \
    -DGPRAT_SYCL_INTEL=$GPRAT_SYCL_INTEL \
    -DHIP_TARGETS=$HIP_TARGETS \
    -DGPRAT_ENABLE_TESTS=ON \
    -DGPRAT_ENABLE_EXAMPLES=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

################################################################################
# Compile code
################################################################################
cmake --build --preset $PRESET -- VERBOSE=1 -j
cmake --install build/$PRESET

################################################################################
# Run tests
################################################################################
cd build/$PRESET
ctest --output-on-failure --no-tests=ignore -C Release -j 2
