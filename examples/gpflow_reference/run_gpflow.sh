#!/bin/bash
# Input $1: Specify cpu/gpu

# Create & Activate python environment
if [ ! -d "gpflow_env" ]; then
    python3 -m venv gpflow_env
fi
source gpflow_env/bin/activate

# install gpflow if not already installed
if ! python3 -c "import gpflow"; then
    pip3 install --no-cache-dir -r requirements.txt
    git clone https://github.com/GPflow/GPflow.git
    cd GPflow
    git checkout v2.9.2
    git apply ../gpflow_mkl.patch
    pip install -e .
    cd ..
fi

# run on CPU or GPU
if [[ "$1" == "gpu" ]]
then
    module load cuda/11.8.0
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
    # Execute the python script
    python3 execute.py --use-gpu
elif [[ "$1" == "cpu" ]]
then
    # Execute the python script
    python3 execute.py
else
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi
