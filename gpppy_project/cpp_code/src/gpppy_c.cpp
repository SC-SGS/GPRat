#include "../include/gp_headers/gpppy_c.hpp"
#include "../include/gp_headers/utils_c.hpp"

#include <stdexcept>
#include <iomanip>
#include <cstdio>
#include <sstream>

namespace gpppy
{
    // Initialize of the Gaussian process data constructor
    GP_data::GP_data(const std::string &f_path, int n)
    {
        n_samples = n;
        file_path = f_path;
        data = utils::load_data(f_path, n);
    }

    // Initialize of the Gaussian process constructor
    GP::GP(std::vector<double> input, std::vector<double> output, int n_tiles, int n_tile_size, double l, double v, double n, int n_r, std::vector<bool> trainable_bool)
    {
        _training_input = input;
        _training_output = output;
        _n_tiles = n_tiles;
        _n_tile_size = n_tile_size;
        lengthscale = l;
        vertical_lengthscale = v;
        noise_variance = n;
        n_regressors = n_r;
        trainable_params = trainable_bool;
    }

    // Return training input data
    std::vector<double> GP::get_training_input() const
    {
        return _training_input;
    }

    // Return training output data
    std::vector<double> GP::get_training_output() const
    {
        return _training_output;
    }

    // Print Gausian process attributes
    std::string GP::repr() const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(12);
        oss << "Kernel_Params: [lengthscale=" << lengthscale
            << ", vertical_lengthscale=" << vertical_lengthscale
            << ", noise_variance=" << noise_variance
            << ", n_regressors=" << n_regressors
            << ", trainable_params l=" << trainable_params[0]
            << ", trainable_params v=" << trainable_params[1]
            << ", trainable_params n=" << trainable_params[2]
            << "]";
        return oss.str();
    }

    // Predict output for test input
    std::vector<double> GP::predict(const std::vector<double> &test_data, int m_tiles, int m_tile_size)
    {
        std::vector<double> result;
	    // result = fut.get();
        hpx::threads::run_as_hpx_thread([this, &result, &test_data, m_tiles, m_tile_size]()
                                       {
                                           result = predict_hpx(_training_input, _training_output, test_data,
                                                                  _n_tiles, _n_tile_size, m_tiles, m_tile_size,
                                                                  lengthscale, vertical_lengthscale, noise_variance, 
                                                                  n_regressors).get(); // Wait for and get the result from the future
                                       });
        return result;
    }

    // Predict output for test input and additionally provide uncertainty for the predictions
    std::vector<std::vector<double>> GP::predict_with_uncertainty(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
    {
        std::vector<std::vector<double>> result;
        hpx::threads::run_as_hpx_thread([this, &result, &test_input, m_tiles, m_tile_size]()
                                        {
                                            result = predict_with_uncertainty_hpx(_training_input, _training_output, test_input,
                                                                                _n_tiles, _n_tile_size, m_tiles, m_tile_size,
                                                                                lengthscale, vertical_lengthscale, noise_variance, 
                                                                                n_regressors).get(); // Wait for and get the result from the future
                                        });
        return result;
    }

    
    // Predict output for test input and additionally provide full posterior covariance matrix
    std::vector<std::vector<double>> GP::predict_with_full_cov(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
    {
        std::vector<std::vector<double>> result;
        hpx::threads::run_as_hpx_thread([this, &result, &test_input, m_tiles, m_tile_size]()
                                        {
                                            result = predict_with_full_cov_hpx(_training_input, _training_output, test_input,
                                                                                _n_tiles, _n_tile_size, m_tiles, m_tile_size,
                                                                                lengthscale, vertical_lengthscale, noise_variance, 
                                                                                n_regressors).get(); // Wait for and get the result from the future
                                        });
        return result;
    }

    // Optimize hyperparameters for a specified number of iterations
    std::vector<double> GP::optimize(const gpppy_hyper::Hyperparameters &hyperparams)
    {

        std::vector<double> losses;
        hpx::threads::run_as_hpx_thread([this, &losses, &hyperparams]()
                                        {
                                            losses = optimize_hpx(_training_input, _training_output, _n_tiles, _n_tile_size,
                                                                   lengthscale, vertical_lengthscale, noise_variance, n_regressors,
                                                                   hyperparams, trainable_params).get(); // Wait for and get the result from the future
                                        });
        return losses;
    }

    // Perform a single optimization step
    double GP::optimize_step(gpppy_hyper::Hyperparameters &hyperparams, int iter)
    {

        double loss;
        hpx::threads::run_as_hpx_thread([this, &loss, &hyperparams, iter]()
                                        {
                                            loss = optimize_step_hpx(_training_input, _training_output, _n_tiles, _n_tile_size,
                                                           lengthscale, vertical_lengthscale, noise_variance, n_regressors,
                                                           hyperparams, trainable_params, iter).get(); // Wait for and get the result from the future
                                        });
        return loss;
    }

    // Calculate loss for given data and Gaussian process model
    double GP::calculate_loss()
    {
        double hyperparameters[3];
        hyperparameters[0] = lengthscale;          // lengthscale
        hyperparameters[1] = vertical_lengthscale; // vertical_lengthscale
        hyperparameters[2] = noise_variance;       // noise_variance

        double loss;
        hpx::threads::run_as_hpx_thread([this, &loss, &hyperparameters]()
                                        {
                                            loss = compute_loss_hpx(_training_input, _training_output, _n_tiles, _n_tile_size,
                                                          n_regressors, hyperparameters).get(); // Wait for and get the result from the future
                                        });
        return loss;
    }

    // Compute Cholesky decomposition
    std::vector<std::vector<double>> GP::cholesky()
    {
        std::vector<std::vector<double>> result;
        hpx::threads::run_as_hpx_thread([this, &result]()
                                        {
                                            result = cholesky_hpx(_training_input, _training_output, _n_tiles, _n_tile_size,
                                                                   lengthscale, vertical_lengthscale, noise_variance, 
                                                                   n_regressors).get(); // Wait for and get the result from the future
                                        });
        return result;
    }

}
