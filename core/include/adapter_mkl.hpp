#ifndef ADAPTER_MKL_H
#define ADAPTER_MKL_H

#include <hpx/future.hpp>
#include <vector>

typedef enum BLAS_TRANSPOSE {Blas_no_trans=111, Blas_trans=112} BLAS_TRANSPOSE;

// =============================================================================
// BLAS operations on CPU with MKL
// =============================================================================

// BLAS operations for tiled cholkesy -------------------------------------- {{{

/**
 * @brief In-place Cholesky decomposition of A
 *
 * @param f_A matrix to be factorized
 * @param N matrix dimension
 * @return factorized, lower triangular matrix f_L
 */
hpx::shared_future<std::vector<double>> potrf(hpx::shared_future<std::vector<double>> f_A, std::size_t N);

/**
 * @brief In-place solve X * L^T = A where L lower triangular
 *
 * @param f_L Cholesky factor matrix
 * @param f_A right hand side matrix
 * @param N matrix dimension
 * @return solution matrix f_X
 */
hpx::shared_future<std::vector<double>>
trsm(hpx::shared_future<std::vector<double>> f_L, hpx::shared_future<std::vector<double>> f_A, std::size_t N);

/**
 * @brief Symmetric rank update: A = A - B * B^T
 * @param f_A Base matrix
 * @param f_B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix f_A
 */
hpx::shared_future<std::vector<double>>
syrk(hpx::shared_future<std::vector<double>> f_A, hpx::shared_future<std::vector<double>> f_B, std::size_t N);

/**
 * @brief General matrix-matrix multiplication: C = C - A * B^T
 * @param f_C Base matrix
 * @param f_B Right update matrix
 * @param f_A Left update matrix
 * @param N matrix dimension
 * @return updated matrix f_X
 */
hpx::shared_future<std::vector<double>> gemm(hpx::shared_future<std::vector<double>> f_A,
                         hpx::shared_future<std::vector<double>> f_B,
                         hpx::shared_future<std::vector<double>> f_C,
                         std::size_t N);

/**
 * @brief In-place solve L(^T) * x = a where L lower triangular
 * @param f_L Cholesky factor matrix
 * @param f_a right hand side vector
 * @param N matrix dimension
 * @param transpose_L
 * @return solution vector f_x
 */
hpx::shared_future<std::vector<double>>
trsv(hpx::shared_future<std::vector<double>> L, hpx::shared_future<std::vector<double>> f_a, std::size_t N, BLAS_TRANSPOSE transpose_L);

/**
 * @brief General matrix-vector multiplication: b = b - A(^T) * a
 * @param f_A update matrix
 * @param f_a update vector
 * @param f_b base vector
 * @param N matrix dimension
 * @param transpose_A
 * @return updated vector f_b
 */
hpx::shared_future<std::vector<double>> gemv(hpx::shared_future<std::vector<double>> f_A,
                           hpx::shared_future<std::vector<double>> f_a,
                           hpx::shared_future<std::vector<double>> f_b,
                           std::size_t N,
                           BLAS_TRANSPOSE transpose_A);

// A = y*beta^T + A
hpx::shared_future<std::vector<double>> ger(hpx::shared_future<std::vector<double>> f_A,
                        hpx::shared_future<std::vector<double>> f_x,
                        hpx::shared_future<std::vector<double>> f_y,
                        std::size_t N);

// C = C + A * B^T
hpx::shared_future<std::vector<double>> gemm_diag(hpx::shared_future<std::vector<double>> f_A,
                              hpx::shared_future<std::vector<double>> f_B,
                              hpx::shared_future<std::vector<double>> f_C,
                              std::size_t N);

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
hpx::shared_future<std::vector<double>> gemv_p(hpx::shared_future<std::vector<double>> f_A,
                           hpx::shared_future<std::vector<double>> f_a,
                           hpx::shared_future<std::vector<double>> f_b,
                           std::size_t N_row,
                           std::size_t N_col);

// }}} ------------------------------- end of BLAS operations for tiled cholkesy

// BLAS operations used in uncertainty computation ------------------------- {{{

// in-place solve X * L = A where L lower triangular
hpx::shared_future<std::vector<double>> trsm_l_KcK(hpx::shared_future<std::vector<double>> L,
                               hpx::shared_future<std::vector<double>> f_A,
                               std::size_t N,
                               std::size_t M);

// C = C - A * B
hpx::shared_future<std::vector<double>> gemm_l_KcK(hpx::shared_future<std::vector<double>> f_A,
                               hpx::shared_future<std::vector<double>> f_B,
                               hpx::shared_future<std::vector<double>> f_C,
                               std::size_t N,
                               std::size_t M);

// C = C - A^T * B
hpx::shared_future<std::vector<double>> gemm_cross_tcross_matrix(hpx::shared_future<std::vector<double>> f_A,
                                             hpx::shared_future<std::vector<double>> f_B,
                                             hpx::shared_future<std::vector<double>> f_C,
                                             std::size_t N,
                                             std::size_t M);

// }}} --------------------------------- end of BLAS for uncertainty computation

// BLAS operations used in optimization step ------------------------------- {{{

// in-place solve L * X = A where L lower triangular
hpx::shared_future<std::vector<double>> trsm_l_matrix(hpx::shared_future<std::vector<double>> L,
                                  hpx::shared_future<std::vector<double>> f_A,
                                  std::size_t N,
                                  std::size_t M);

// C = C - A * B
hpx::shared_future<std::vector<double>> gemm_l_matrix(hpx::shared_future<std::vector<double>> f_A,
                                  hpx::shared_future<std::vector<double>> f_B,
                                  hpx::shared_future<std::vector<double>> f_C,
                                  std::size_t N,
                                  std::size_t M);

// in-place solve L^T * X = A where L upper triangular
hpx::shared_future<std::vector<double>> trsm_u_matrix(hpx::shared_future<std::vector<double>> L,
                                  hpx::shared_future<std::vector<double>> f_A,
                                  std::size_t N,
                                  std::size_t M);

// C = C - A^T * B
hpx::shared_future<std::vector<double>> gemm_u_matrix(hpx::shared_future<std::vector<double>> f_A,
                                  hpx::shared_future<std::vector<double>> f_B,
                                  hpx::shared_future<std::vector<double>> f_C,
                                  std::size_t N,
                                  std::size_t M);

// Dot product used in dot calculation
double dot(std::size_t N, std::vector<double> a, std::vector<double> b);

// C = C - A * B
hpx::shared_future<std::vector<double>> dot_uncertainty(hpx::shared_future<std::vector<double>> f_A,
                                    hpx::shared_future<std::vector<double>> f_R,
                                    std::size_t N,
                                    std::size_t M);

// C = C - A * B
hpx::shared_future<std::vector<double>> gemm_grad(hpx::shared_future<std::vector<double>> f_A,
                              hpx::shared_future<std::vector<double>> f_B,
                              hpx::shared_future<std::vector<double>> f_R,
                              std::size_t N,
                              std::size_t M);

// }}} --------------------------------------- end of BLAS for optimization step

#endif  // end of ADAPTER_MKL_H
