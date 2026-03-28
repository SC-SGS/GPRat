'''
Reference implementation for Gaussian process regression using GPyTorch.
'''

# Imports
import time
import logging
import torch
import gpytorch
import os
import argparse
from config import get_config
from gpytorch_logger import setup_logging
from utils import load_data, ExactGPModel, train, predict, predict_with_var
import gc

# Global definitions
logger = logging.getLogger()
log_filename = "./gpytorch_logs.log"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Flag to use GPU (assuming available)",
)

args = parser.parse_args()

def get_device(use_gpu):
    """
    Returns a tuple containing 
    - a device object corrsponding to user preference if available, and the CPU
      otherwise
    - a string representation of the device

    Args:
        use_gpu (bool): whether to use GPU or not if available

    Returns:
        tuple: a tuple of a device and a string
    """
    if not use_gpu:
        return torch.device("cpu"), "cpu"

    # NVIDIA CUDA or AMD ROCm
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)

    # Intel GPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), torch.xpu.get_device_name(0)

    return torch.device("cpu"), "cpu"


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "xpu":
        torch.xpu.synchronize()


def gpytorch_run(
        config, output_file, size_train, size_test, loop_index, cores, device, target, \
        is_warmup=False
        ):
    """
    Run the Gaussian process regression pipeline.

    Args:
        config (dict):                  Configuration parameters for the pipeline.
        output_csv_obj (csv.writer):    CSV writer object for writing output data.
        size_train (int):               Size of the training dataset.
        size_test (int):                Size of the test dataset.
        loop_index (int):               Loop index.
        cores (int):                    Number of CPU cores to use.
        device (torch.device):          Device to use for computation.
        target (str):                   Target device name.
        is_warmup (bool):               Flag to indicate if this is a warmup run.
    """
    total_t = time.perf_counter()

    # Load data
    load_t = time.perf_counter()
    X_train, Y_train, X_test, Y_test = load_data(
        train_in_path=config["TRAIN_IN_FILE"],
        train_out_path=config["TRAIN_OUT_FILE"],
        test_in_path=config["TEST_IN_FILE"],
        test_out_path=config["TEST_OUT_FILE"],
        size_train=size_train,
        size_test=size_test,
        n_regressors=config["N_REG"]
    )
    if args.use_gpu and device.type != "cpu":
        X_train, Y_train, X_test, Y_test = \
            X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)
        sync_if_needed(device)
    load_t = time.perf_counter() - load_t

    # Initialize model
    init_t = time.perf_counter()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 0.1
    model = ExactGPModel(X_train, Y_train, likelihood)
    if args.use_gpu and device.type != "cpu":
        model = model.to(device)
        likelihood = likelihood.to(device)
        sync_if_needed(device)
    init_t = time.perf_counter() - init_t

    # Train model
    train_t = time.perf_counter()
    train(model, likelihood, X_train, Y_train, training_iter=config['OPT_ITER'])
    sync_if_needed(device)
    train_t = time.perf_counter() - train_t

    # Make predictions with uncertainty
    pred_var_t = time.perf_counter()
    f_pred, f_var = predict_with_var(model, likelihood, X_test)
    sync_if_needed(device)
    pred_var_t = time.perf_counter() - pred_var_t

    # Make predictions without uncertainty
    pred_t = time.perf_counter()
    f_pred = predict(model, likelihood, X_test)
    sync_if_needed(device)
    pred_t = time.perf_counter() - pred_t

    # Assign runtimes
    TOTAL_TIME = time.perf_counter() - total_t
    LOAD_TIME = load_t
    INIT_TIME = init_t
    OPT_TIME = train_t
    PRED_UNCER_TIME = pred_var_t
    PREDICTION_TIME = pred_t

    if not is_warmup:

        row_data = \
            f"{target},{cores},{size_train},{size_test},{config['N_REG']},"\
            f"{config['OPT_ITER']},{TOTAL_TIME},{LOAD_TIME},{INIT_TIME},{OPT_TIME},"\
            f"{PRED_UNCER_TIME},{PREDICTION_TIME},{loop_index}\n"
        output_file.write(row_data)

        logger.info(
            f"{target},{cores},{size_train},{size_test},{config['N_REG']},"\
            f"{config['OPT_ITER']},{TOTAL_TIME},{LOAD_TIME},{INIT_TIME},{OPT_TIME},"\
            f"{PRED_UNCER_TIME},{PREDICTION_TIME},{loop_index}\n"
        )


def execute():
    """
    This function performs following steps:
        - Set up logging.
        - Load configuration file.
        - Write header for the output CSV file.
        - Set up PyTorch configurations based on the loaded config.
        - Iterate through different training sizes and for each training size
        loop for a specified amount of times while executing `gpytorch_run` function.
    """

    torch.set_num_interop_threads(1)

    setup_logging(log_filename, True, logger)
    logger.info("\n")
    logger.info("-" * 40)
    logger.info("Load config file.")
    config = get_config()

    file_path = "./output.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a") as output_file:
        if not file_exists or os.stat(file_path).st_size == 0:
            logger.info(
                "Target,Cores,N_train,N_test,N_regressor,Opt_iter,Total_time,Load_time,"\
                "Init_time,Opt_Time,Pred_Uncer_time,Predict_time,N_loop"
            )
            header = \
                "Target,Cores,N_train,N_test,N_regressor,Opt_iter,Total_time,Load_time,"\
                "Init_time,Opt_Time,Pred_Uncer_time,Predict_time,N_loop\n"
            output_file.write(header)

        if config["PRECISION"] == "float128":
            torch.set_default_dtype(torch.float128)
        else:
            torch.set_default_dtype(torch.float64)

        device, target = get_device(args.use_gpu)
        test_scale_factor = config["STEP"] if config["SCALE_TEST_WITH_TRAIN"] else 1

        gpytorch_run(config, output_file, config["TRAIN_SIZE_END"], \
                     config["TRAIN_SIZE_END"], 0, config["END_CORES"], device, \
                     target, True)

        # CPU
        if device.type == "cpu":

            cores = config["START_CORES"]

            while cores <= config["END_CORES"]:

                torch.set_num_threads(cores)

                data_size = config["TRAIN_SIZE_START"]
                test_size = config["TEST_SIZE"] if not config["SCALE_TEST_WITH_TRAIN"] \
                    else config["TRAIN_SIZE_START"]

                # Loop over training data sizes
                while data_size <= config["TRAIN_SIZE_END"]:

                    # Loop over different test iterations
                    for loop_index in range(config["LOOP"]):

                        # Loop to create test runs
                        logger.info("*" * 40)
                        logger.info(
                            f"Cores: {cores}, Train Size: {data_size}, Loop: {loop_index}"
                            )
                        gc.collect()
                        gpytorch_run(
                            config, output_file, data_size, test_size, loop_index, cores, 
                            device, target
                        )

                    # Update sizes
                    data_size = data_size * config["STEP"]
                    test_size = test_size * test_scale_factor
                
                cores *= 2

        # GPU
        else:

            torch.set_num_threads(1)
            
            # Set train and test sizes
            data_size = config["TRAIN_SIZE_START"]
            test_size = config["TEST_SIZE"] if not config["SCALE_TEST_WITH_TRAIN"] \
                else config["TRAIN_SIZE_START"]

            # Loop over training data sizes
            while data_size <= config["TRAIN_SIZE_END"]:

                # Loop over different test iterations
                for loop_index in range(config["LOOP"]):

                    print(f"Before cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"After cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

                    # Loop to create test runs
                    logger.info("*" * 40)
                    logger.info(f"Cores: {1}, Train Size: {data_size}, Loop: {loop_index}")
                    gpytorch_run(
                        config, output_file, data_size, test_size, loop_index, 1,
                        device, target
                    )

                # Update sizes
                data_size = data_size * config["STEP"]
                test_size = test_size * test_scale_factor

        logger.info("Completed the program.")


def is_mkl_enabled():
    torch_config = torch.__config__.show()
    index = torch_config.index("USE_MKL")
    value = torch_config[index+8:index+10]
    return True if value == 'ON' else False


if __name__ == "__main__":
    setup_logging(log_filename, True, logger)
    print("","-" * 18, "\n", "MKL enabled:", is_mkl_enabled(), "\n", "-" * 18)
    execute()
