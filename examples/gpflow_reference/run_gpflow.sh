#!/bin/bash
# Input $1: Specify cpu/gpu/arm
# Input $2: Specify nvidia/amd/intel (only necessary if gpu is specified)

if [[ "$1" == "gpu" ]] #############################################################################
then

    # Intel packages require more recent Python, e.g. 3.12
    if [[ "$2" == "intel" ]]; then
        module load python/3.12
        echo "intel"
    fi

    # Create Python environment if not present
    if [ ! -d "gpflow_gpu_env" ]; then
        python -m venv gpflow_gpu_env --clear
    fi

    # Activate Python environment and take measures to avoid instant catastrophy
    source gpflow_gpu_env/bin/activate
    python -m ensurepip --upgrade
    pip install --upgrade pip
    pip install setuptools==80.0.0

    # Install gpflow if not already installed
    if ! python -c "import gpflow"; then

        if [[ "$2" == "nvidia" ]]; then ###########################################################

            module load cuda/12.0.1

            pip install --no-cache-dir tensorflow[and-cuda] gpflow==2.9.2

            export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

        elif [[ "$2" == "amd" ]]; then ############################################################
           
            module load rocm

            pip install --no-cache-dir tensorflow-rocm==2.19.1 \
            -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/

            pip install --no-cache-dir gpflow tensorboard~=2.19.0 --timeout 600

        elif [[ "$2" == "intel" ]]; then ##########################################################

            python -m pip install --no-cache-dir intel-tensorflow gpflow

        else ######################################################################################

            echo "Please specify gpu type: nvidia/amd/intel"
            exit 1

        fi ########################################################################################

    fi

    # Run on GPU
    python execute.py --use-gpu

elif [[ "$1" == "cpu" ]]
then

    module load python/3.10.16
    # Create & Activate python environment
    if [ ! -d "gpflow_cpu_env" ]; then
        python -m venv gpflow_cpu_env
    fi
    source gpflow_cpu_env/bin/activate
    # Install gpflow if not already installed
    if ! python -c "import gpflow"; then
        pip install --no-cache-dir -r requirements_cpu.txt
        # manually install GPflow
        git clone https://github.com/GPflow/GPflow.git
        cd GPflow
        git checkout v2.10.0
        git apply ../gpflow_mkl.patch
        pip install -e .
        cd ..
    fi
    # Run on CPU
    python execute.py

elif [[ "$1" == "arm" ]]
then

    spack load python@3.10
    # Create & Activate python environment
    if [ ! -d "gpflow_arm_env" ]; then
        python -m venv gpflow_arm_env
    fi
    source gpflow_arm_env/bin/activate
    # Install gpflow if not already installed
    if ! python -c "import gpflow"; then
        pip install --no-cache-dir -r requirements_gpu.txt
    fi
    # Run on ARM
    python execute.py

else

    echo "Please specify input parameter: cpu/gpu/arm"
    exit 1

fi
