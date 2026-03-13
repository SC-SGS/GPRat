#!/bin/bash
# Input $1: Specify cpu/gpu/arm
# Input $2: Specify nvidia/amd/intel (only necessary if gpu is specified)

if [[ "$1" == "gpu" ]] #############################################################################
then

    # Intel packages require more recent Python, e.g. 3.12
    # if [[ "$2" == "intel" ]]; then
    #     module load python/3.12
    #     echo "intel"
    # fi

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

            pip freeze > requirements_gpflow_nvidia.txt

        elif [[ "$2" == "amd" ]]; then ############################################################

            pip install --no-cache-dir tensorflow-probability[tf]==0.24.0 tensorboard==2.17 ml-dtypes==0.3.1 --timeout 600

            pip install --no-cache-dir gpflow==2.9.2

            pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu

            pip install tensorflow-rocm==2.17.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/

            pip freeze > requirements_gpflow_amd.txt

        elif [[ "$2" == "intel" ]]; then ##########################################################

            module load python/3.10.16

            # Very important
            source /opt/intel/oneapi/setvars.sh

            # First, let pip install GPflow and cause some havoc. We'll fix it later.
            pip install --no-cache-dir gpflow tensorflow-probability~=0.23.0
            
            # Whatever pip installs here is almost guaranteed to fail, so away with it.
            pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu

            # Install a TensorFlow version that matches what Intel expects
            pip install --no-cache-dir tensorflow==2.15.1

            # Install Intel extensions for TensorFlow
            pip install --no-cache-dir --upgrade intel-extension-for-tensorflow[xpu]

            # Install setuptools because something keeps overwriting it
            pip install setuptools==78.0.0

            pip freeze > requirements_gpflow_intel.txt

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
