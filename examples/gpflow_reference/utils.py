import numpy as np
import gpflow
import tensorflow as tf


def generate_regressor(x_original, n_regressors):
    """
    Generate regressor matrix by padding the original input array with zeros 
    from the left for (n_regressors - 1) positions, and rolling over the input
    array where the window size equals n_regressors.

    Args:
        x_original (array-like): The original input array.
        n_regressors (int): The number of regressors.

    Returns:
        array-like: The regressor matrix.
    """
    X = []
    x_padded = np.pad(
        x_original,
        pad_width=(n_regressors - 1, 0),
        mode="constant",
        constant_values=(0),
    )

    for _ in range(len(x_original)):
        X.append(x_padded[:n_regressors])
        x_padded = np.roll(x_padded, -1)

    X = np.array(X)

    return X


def load_data(
    train_in_path,
    train_out_path,
    test_in_path,
    test_out_path,
    size_train: int,
    size_test: int,
    n_regressors: int,
):
    """
    Load and preprocess data for Gaussian process regression.

    Args:
        train_in_path (str): Path to the training input data file.
        train_out_path (str): Path to the training output data file.
        test_in_path (str): Path to the testing input data file.
        test_out_path (str): Path to the testing output data file.
        size_train (int): Size of the training dataset.
        size_test (int): Size of the testing dataset.
        n_regressors (int): Number of regressors.

    Returns:
        tuple: A tuple containing:
            - X_train (numpy.ndarray): Regressor matrix for training data.
            - Y_train (numpy.ndarray): Target values for training data.
            - X_test (numpy.ndarray): Regressor matrix for testing data.
            - Y_test (numpy.ndarray): Target values for testing data.
    """
    x_train_in = np.loadtxt(train_in_path, dtype="d")[:size_train]
    x_test_in = np.loadtxt(test_in_path, dtype="d")[:size_test]

    X_train = generate_regressor(x_train_in, n_regressors).astype("d")
    X_test = generate_regressor(x_test_in, n_regressors).astype("d")

    Y_train = np.loadtxt(train_out_path, dtype="d")[:size_train, None]
    Y_test = np.loadtxt(test_out_path, dtype="d")[:size_test, None]

    return X_train, Y_train, X_test, Y_test


def init_model(X, Y, k_var=1.0, k_lscale=1.0,  noise_var=0.1, params_summary: bool = False):
    """
    Train a Gaussian process regression model using GPflow.

    Args:
        X (numpy.ndarray): The input data matrix.
        Y (numpy.ndarray): The target data vector.
        k_var (float, optional): Variance parameter of the kernel. Defaults to 1.0.
        k_lscale (float, optional): Lengthscale parameter of the kernel. Defaults to 1.0.
        noise_var (float, optional): Noise variance parameter. Defaults to 0.1.
        params_summary (bool, optional): Whether to print a summary of model parameters. Defaults to False.

    Returns:
        gpflow.models.GPR: The trained Gaussian process regression model.
    """
    model = gpflow.models.GPR(
        (X, Y),
        kernel=gpflow.kernels.SquaredExponential(
            variance=k_var,
            lengthscales=k_lscale,
        ),
        # likelihood = gpflow.likelihoods.Gaussian(0.1),
        noise_variance=noise_var,
    )

    if params_summary:
        gpflow.utilities.print_summary(model)

    return model


def optimize_model(model, training_iter):
    """
    Optimize the parameters of the given GPflow model.

    Args:
        model (gpflow.models.GPModel): The GPflow model to be optimized.

    Returns:
        None
    """
    opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    @tf.function
    def optimization_step(): 
        with tf.GradientTape() as tape:
            # Compute the loss inside the tape context
            loss = model.training_loss()
        # Compute the gradients of the loss with respect to the model's trainable variables
        gradients = tape.gradient(loss, model.trainable_variables)
        # Apply the gradients to update the model's trainable variables
        opt.apply_gradients(zip(gradients, model.trainable_variables))

    # Run the optimization step
    for i in range(training_iter): 
        optimization_step()

    return None


def predict_with_full_cov(model, X_test):
    """
    Predict latent function values and observed target values for the given test data.

    Args:
        model (gpflow.models.GPModel): The trained GPflow model.
        X_test (numpy.ndarray): The test input data.

    Returns:
        f_pred (numpy.ndarray): Mean of latent function values for test data.
        f_var_matrix (numpy.ndarray): Full covariance of latent function values for test data.
    """
    f_pred, f_var_matrix = model.predict_f(X_test, full_cov=True)
    
    return f_pred, f_var_matrix


def predict_with_var(model, X_test):
    """
    Predict latent function values and observed target values for the given test data.

    Args:
        model (gpflow.models.GPModel): The trained GPflow model.
        X_test (numpy.ndarray): The test input data.

    Returns:
        f_pred (numpy.ndarray): Mean of latent function values for test data.
        f_var (numpy.ndarray): Variance of latent function values for test data.
    """
    f_pred, f_var = model.predict_f(X_test)
    
    return f_pred, f_var


def predict(model, X_test):
    """
    Predict latent function values and observed target values for the given test data.

    Args:
        model (gpflow.models.GPModel): The trained GPflow model.
        X_test (numpy.ndarray): The test input data.

    Returns:
        f_pred (numpy.ndarray): Mean of latent function values for test data.
    """
    f_pred = model.predict_f(X_test)
    
    return f_pred


def calculate_error(Y_test, Y_pred):
    """
    Calculate the error between the true target values and the predicted target values.

    Args:
        Y_test (numpy.ndarray): True target values.
        Y_pred (numpy.ndarray): Predicted target values.

    Returns:
        float: The error between true and predicted target values.
    """
    return np.linalg.norm(Y_test - Y_pred)
