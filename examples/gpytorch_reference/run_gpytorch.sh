#!/bin/bash
# Input $1: Specify cpu/gpu

# Create & Activate python enviroment
if [ ! -d "gpytorch_env" ]; then
    python -m venv gpytorch_env
fi

# Activate enviroment
source gpytorch_env/bin/activate

# Install requirements
if ! python -c "import gpytorch"; then
    pip install gpytorch==1.13
fi

# run on CPU or GPU
if [[ "$1" == "gpu" ]]
then
    # Execute the python script
    python execute.py --use-gpu
elif [[ "$1" == "cpu" ]]
then
    # Execute the python script
    python execute.py
else
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi
