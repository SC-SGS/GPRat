'''
Reference implementation for Gaussian process regression using GPflow.
'''

# Imports
import argparse
import gc
import logging
import os
import time
import numpy as np
from config import get_config
from utils import (
    init_model,
    load_data,
    optimize_model,
    predict,
    predict_with_full_cov,
    predict_with_var,
)
import tensorflow as tf
import tensorflow.python.util._pywrap_util_port as tf_util
from tensorflow.python.eager import context
import gpflow
from gpflow_logger import setup_logging

# Global definitions
logger = logging.getLogger()
log_filename = "./gpflow_logs.log"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Flag to use GPU (assuming available)",
)
args = parser.parse_args()

# Environment variables
if not args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_OVERRIDE_GLOBAL_THREADPOOL"] = "1"


def sync_if_needed(is_cuda_gpu):
    if is_cuda_gpu:
        tf.test.experimental.sync_devices()


def gpflow_run(target, is_cuda_gpu, config, output_file, size_train, size_test, \
               loop_index, cores, is_warmup=False):
    """
    Run the GPflow regression pipeline.

    Args:
        target (str):       String of target (cpu/gpu) for logs.
        is_cuda_gpu (bool): Whether CUDA GPU is being used or not.
        config (dict):      Configuration parameters for the pipeline.
        output_csv_obj (csv.writer): CSV writer object for writing output data.
        size_train (int):   Size of the training dataset.
        size_test (int):    Size of the test dataset.
        loop_index (int):   Index for the current loop iteration.
        cores (int):        Number of CPU cores to use.
        is_warmup (bool):   Flag to indicate if this is a warmup run.
    """
    total_t = time.perf_counter()

    load_t = time.perf_counter()
    X_train, Y_train, X_test, Y_test = load_data(
        train_in_path=config["TRAIN_IN_FILE"],
        train_out_path=config["TRAIN_OUT_FILE"],
        test_in_path=config["TEST_IN_FILE"],
        test_out_path=config["TEST_OUT_FILE"],
        size_train=size_train,
        size_test=size_test,
        n_regressors=config["N_REG"],
    )
    load_t = time.perf_counter() - load_t

    init_t = time.perf_counter()
    model = init_model(
        X_train,
        Y_train,
        k_var=1.0,
        k_lscale=1.0,
        noise_var=0.1,
        params_summary=False,
    )
    init_t = time.perf_counter() - init_t

    opti_t = time.perf_counter()
    optimize_model(model, training_iter=config["OPT_ITER"])
    sync_if_needed(is_cuda_gpu)
    opti_t = time.perf_counter() - opti_t

    pred_full_t = time.perf_counter()
    f_pred_full, f_var_full = predict_with_full_cov(model, X_test)
    sync_if_needed(is_cuda_gpu)
    pred_full_t = time.perf_counter() - pred_full_t

    pred_var_t = time.perf_counter()
    f_pred, f_var = predict_with_var(model, X_test)
    sync_if_needed(is_cuda_gpu)
    pred_var_t = time.perf_counter() - pred_var_t

    pred_t = time.perf_counter()
    f_pred = predict(model, X_test)
    sync_if_needed(is_cuda_gpu)
    pred_t = time.perf_counter() - pred_t

    TOTAL_TIME = time.perf_counter() - total_t
    LOAD_TIME = load_t
    INIT_TIME = init_t
    OPT_TIME = opti_t
    PRED_FULL_TIME = pred_full_t
    PRED_UNCER_TIME = pred_var_t
    PREDICTION_TIME = pred_t

    if not is_warmup:

        row_data = \
            f"{target},{cores},{size_train},{size_test},{config['N_REG']},"\
            f"{config['OPT_ITER']},{TOTAL_TIME},{LOAD_TIME},{INIT_TIME},{OPT_TIME},"\
            f"{PRED_FULL_TIME},{PRED_UNCER_TIME},{PREDICTION_TIME},{loop_index}\n"
        output_file.write(row_data)

        logger.info(
            f"{target},{cores},{size_train},{size_test},{config['N_REG']},"\
            f"{config['OPT_ITER']},{TOTAL_TIME},{LOAD_TIME},{INIT_TIME},{OPT_TIME},"\
            f"{PRED_FULL_TIME},{PRED_UNCER_TIME},{PREDICTION_TIME},{loop_index}"
        )


def execute():
    """
    Execute the main process:
        - Set up logging.
        - Load configuration file.
        - Initialize output CSV file.
        - Write header to the output CSV file.
        - Set up TensorFlow and GPflow configurations based on the loaded config.
        - Iterate through different training sizes and for each training size
        loop for a specified amount of times while executing `gpflow_run` function.
    """

    # Init
    config = get_config()
    if config["PRECISION"] == "float32":
        gpflow.config.set_default_float(np.float32)
    else:
        gpflow.config.set_default_float(np.float64)
    test_scale_factor = config["STEP"] if config["SCALE_TEST_WITH_TRAIN"] else 1
    
    # Check whether TensorFlow is using GPU
    is_cuda_gpu = False
    gpu_devices = tf.config.list_physical_devices("GPU")
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    xpu_devices = tf.config.list_physical_devices("XPU")
    if gpu_devices:
        logger.info(f"GPUs available: {gpu_devices}")
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        target = details['device_name']
        is_cuda_gpu = True
    elif xpu_devices:
        logger.info(f"XPUs available: {xpu_devices}")
        target = "xpu"
    else:
        logger.info("No GPUs/XPUs found. Using CPU.")
        target = "cpu"

    # Output CSV file setup
    file_path = "./output.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a") as output_file:

        # If CSV file non-existent or empty, create/write header
        if not file_exists or os.stat(file_path).st_size == 0:
            logger.info(
                "Target,Cores,N_train,N_test,N_regressor,Opt_iter,Total_time,Load_time,"\
                "Init_time,Opt_Time,Pred_Full_time,Pred_Uncer_time,Predict_time,N_loop"
            )
            header = \
                "Target,Cores,N_train,N_test,N_regressor,Opt_iter,Total_time,Load_time,"\
                "Init_time,Opt_Time,Pred_Full_time,Pred_Uncer_time,Predict_time,N_loop\n"
            output_file.write(header)

        gpflow_run(target, is_cuda_gpu, config, output_file, config["TRAIN_SIZE_END"], \
                   config["TRAIN_SIZE_END"],0, config["END_CORES"], is_warmup=True)

        cores = config["START_CORES"]

        while cores <= config["END_CORES"]:

            data_size = config["TRAIN_SIZE_START"]
            test_size = config["TEST_SIZE"] if not config["SCALE_TEST_WITH_TRAIN"] \
                else config["TRAIN_SIZE_START"]

            context._reset_context()
            tf.config.threading.set_intra_op_parallelism_threads(cores)
            tf.config.threading.set_inter_op_parallelism_threads(1)

            while data_size <= config["TRAIN_SIZE_END"]:

                for loop_index in range(config["LOOP"]):

                    logger.info("*" * 40)
                    gc.collect()
                    gpflow_run(
                        target, is_cuda_gpu, config, output_file, data_size,\
                        test_size,loop_index, cores
                    )

                # Update sizes
                data_size = data_size * config["STEP"]
                test_size = test_size * test_scale_factor

            cores = cores * 2
        
    logger.info("Completed the program.")


def is_mkl_enabled():
    return tf_util.IsMklEnabled()


if __name__ == "__main__":
    setup_logging(log_filename, True, logger)
    print("","-" * 18, "\n", "MKL enabled:", is_mkl_enabled(), "\n", "-" * 18)
    execute()
