source ~/Setup_Scripts/setup_breyerml.sh

if [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then

    ./execute-benchmark.sh yes yes yes yes gpu nvidia cuda
    ./execute-benchmark.sh no no yes yes gpu nvidia sycl

elif [[ "$HOSTNAME" == "simcl1n3" ]]; 
then

    ./execute-benchmark.sh yes yes yes yes gpu amd

elif [[ "$HOSTNAME" == "simcl1n4" ]]; 
then

    ./execute-benchmark.sh yes yes yes yes cpu cpu

fi