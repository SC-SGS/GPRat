#ifndef SYCL_GP_UNCERTAINTY_H
#define SYCL_GP_UNCERTAINTY_H

namespace gprat::sycl_backend
{

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Diagonal elements matrix A
 * @param B Diagonal elements matrix B
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
double *diag_posterior(double *A, double *B, std::size_t M);

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Posterior covariance matrix
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
double *diag_tile(double *A, std::size_t M);

}  // end of namespace gprat::sycl_backend

#endif  // end of SYCL_GP_UNCERTAINTY_H
