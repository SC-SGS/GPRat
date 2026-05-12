#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Input parameter is missing. Using default: Run computations on CPU"
elif [[ "$1" == "cuda" ]]; then
    GPU="--use_cuda"
elif [[ "$1" == "sycl" ]]; then
	GPU="--use_sycl"
elif [[ "$1" != "cpu" ]]; then
    echo "Please specify input parameter: cpu/cuda/sycl"
    exit 1
fi

# Set Spack if on simcl1n1, simcl1n2, simcl1n3, or simcl1n4
if [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" || "$HOSTNAME" == "simcl1n3" || "$HOSTNAME" == "simcl1n4" ]]; then

	spack_destination="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
	source $spack_destination/spack/share/spack/setup-env.sh

fi

if [[ "$2" == "nvidia" ]]; then

	LD_LIBRARY_PATH=/scratch-simcl1/grafml/Programs/oneMath_nvidia/oneMath/install/lib/:$LD_LIBRARY_PATH

elif [[ "$2" == "amd" ]]; then

	LD_LIBRARY_PATH=/scratch-simcl1/grafml/Programs/oneMath_amd/oneMath/install/lib/:$LD_LIBRARY_PATH

elif [[ "$2" == "intel" ]]; then

	echo "The Intel setup is not supported yet." 1>&2
	exit 1
	
fi

if [[ $(hostname -s) == "sven0" || $(hostname -s) == "sven1" ]]; then

	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/hpx/lib64:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/boost/lib:$LD_LIBRARY_PATH
	export LD_PRELOAD=$HOME/git_workspace/build-scripts/build/jemalloc/lib/libjemalloc.so.2

elif [[ $(hostname) == "simcl1n1" || $(hostname) == "simcl1n2" || $(hostname) == "simcl1n3" ]]; then

	# Check if the gprat_gpu_clang environment exists

	if spack env list | grep -q "gprat_gpu_clang"; then

	    echo "Found gprat_gpu_clang environment, activating it."
	    spack env activate gprat_gpu_clang
	    LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
	    LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
        LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH

		if [[ "$1" == "cuda" ]]; then

			module load cuda/12.0.1

		fi
			
	fi

elif [[ $(hostname) == "pcsgs04" ]]; then

	echo "The Intel setup is not supported yet." 1>&2
	exit 1

fi

python3 execute.py $GPU
