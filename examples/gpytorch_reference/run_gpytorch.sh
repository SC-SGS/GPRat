#!/bin/bash
# Input $1: Specify cpu/gpu/arm
# Input $2: Specify nvidia/amd/intel (only necessary if gpu is specified)
# Input $3: number of iterations per config

if [[ "$1" == "gpu" ]]
then
    # Create & Activate python enviroment
    if [ ! -d "gpytorch_gpu_env" ]; then
        python -m venv gpytorch_gpu_env
    fi
    # Activate enviroment
    source gpytorch_gpu_env/bin/activate
    python -m ensurepip --upgrade

    # Install requirements
    if ! python -c "import gpytorch"; then

        pip install --upgrade pip setuptools wheel

        if [[ "$2" == "nvidia" ]]; then ###########################################################

            pip3 install --no-cache-dir torch torchvision \
                --index-url https://download.pytorch.org/whl/cu126

            pip freeze > requirements/requirements_gpytorch_nvidia.txt

        elif [[ "$2" == "amd" ]]; then ############################################################

            pip3 install --no-cache-dir torch torchvision \
                --index-url https://download.pytorch.org/whl/rocm6.4
            
            pip freeze > requirements/requirements_gpytorch_amd.txt

        elif [[ "$2" == "intel" ]]; then ##########################################################

            export PYTORCH_DEBUG_XPU_FALLBACK=1

            # Careful: Intel pulls its own SYCL installation here, make sure no other is loaded!
            pip install --no-cache-dir torch torchvision \
                --index-url https://download.pytorch.org/whl/xpu

            pip freeze > requirements/requirements_gpytorch_intel.txt

        else ######################################################################################

            echo "Please specify gpu type: nvidia/amd/intel"
            exit 1

        fi ########################################################################################

        pip install gpytorch==1.13

    fi

    # Execute the python script
    for n in 0 1 2 3 4
    do
        python execute.py --use-gpu --iteration=$n
    done

elif [[ "$1" == "cpu" ]]
then

    # Create & Activate python enviroment
    if [ ! -d "gpytorch_cpu_env" ]; then
        python -m venv gpytorch_cpu_env
    fi
    # Activate enviroment
    source gpytorch_cpu_env/bin/activate
    # Install requirements
    if ! python -c "import gpytorch"; then
        pip install gpytorch==1.13
        pip freeze > requirements/requirements_gpytorch_cpu.txt
    fi
    # Execute the python script
        for n in 0 1 2 3 4
    do
        python execute.py --iteration=$n
    done

elif [[ "$1" == "arm" ]]
then

    spack load python@3.10
    # Create & Activate python enviroment
    if [ ! -d "gpytorch_arm_env" ]; then
        python -m venv gpytorch_arm_env
    fi
    # Activate enviroment
    source gpytorch_arm_env/bin/activate
    # Install requirements
    if ! python -c "import gpytorch"; then
        pip install gpytorch==1.13
        pip freeze > requirements/requirements_gpytorch_arm.txt
    fi
    # Execute the python script
    for n in 0 1 2 3 4
    do
        python execute.py --iteration=$n
    done

else

    echo "Please specify input parameter: cpu/gpu/arm"
    exit 1

fi
