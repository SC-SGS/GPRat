#ifndef ADAPTER_MKL_H
#define ADAPTER_MKL_H

#include <vector>

// =============================================================================
// BLAS operations on CPU with MKL
// =============================================================================

// BLAS operations for tiled cholkesy -------------------------------------- {{{

/**
 * @brief In-place Cholesky decomposition of A
 *
 * @param A matrix to be factorized
 * @param N size of the matrix
 * @return factorized, lower triangular matrix L
 */
std::vector<double> potrf(std::vector<double> A, int N);

// in-place solve X * L^T = A where L lower triangular
std::vector<double>
trsm(std::vector<double> L, std::vector<double> A, int N);

// A = A - B * B^T
std::vector<double>
syrk(std::vector<double> A, std::vector<double> B, int N);

// C = C - A * B^T
std::vector<double> gemm(std::vector<double> A,
                         std::vector<double> B,
                         std::vector<double> C,
                         int N);

// in-place solve L * x = a where L lower triangular
std::vector<double>
trsv_l(std::vector<double> L, std::vector<double> a, int N);

// b = b - A * a
std::vector<double> gemv_l(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           int N);

// in-place solve L^T * x = a where L lower triangular
std::vector<double>
trsv_u(std::vector<double> L, std::vector<double> a, int N);

// b = b - A^T * a
std::vector<double> gemv_u(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           int N);

// A = y*beta^T + A
std::vector<double> ger(std::vector<double> A,
                        std::vector<double> x,
                        std::vector<double> y,
                        int N);

// C = C + A * B^T
std::vector<double> gemm_diag(std::vector<double> A,
                              std::vector<double> B,
                              std::vector<double> C,
                              int N);

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
std::vector<double> gemv_p(std::vector<double> A,
                           std::vector<double> a,
                           std::vector<double> b,
                           int N_row,
                           int N_col);

// }}} ------------------------------- end of BLAS operations for tiled cholkesy

// BLAS operations used in uncertainty computation ------------------------- {{{

// in-place solve X * L = A where L lower triangular
std::vector<double> trsm_l_KcK(std::vector<double> L,
                               std::vector<double> A,
                               int N,
                               int M);

// C = C - A * B
std::vector<double> gemm_l_KcK(std::vector<double> A,
                               std::vector<double> B,
                               std::vector<double> C,
                               int N,
                               int M);

// C = C - A^T * B
std::vector<double> gemm_cross_tcross_matrix(std::vector<double> A,
                                             std::vector<double> B,
                                             std::vector<double> C,
                                             int N,
                                             int M);

// }}} --------------------------------- end of BLAS for uncertainty computation

// BLAS operations used in optimization step ------------------------------- {{{

// in-place solve L * X = A where L lower triangular
std::vector<double> trsm_l_matrix(std::vector<double> L,
                                  std::vector<double> A,
                                  int N,
                                  int M);

// C = C - A * B
std::vector<double> gemm_l_matrix(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> C,
                                  int N,
                                  int M);

// in-place solve L^T * X = A where L upper triangular
std::vector<double> trsm_u_matrix(std::vector<double> L,
                                  std::vector<double> A,
                                  int N,
                                  int M);

// C = C - A^T * B
std::vector<double> gemm_u_matrix(std::vector<double> A,
                                  std::vector<double> B,
                                  std::vector<double> C,
                                  int N,
                                  int M);

// Dot product used in dot calculation
double dot(int N, std::vector<double> A, std::vector<double> B);

// C = C - A * B
std::vector<double> dot_uncertainty(std::vector<double> A,
                                    std::vector<double> R,
                                    int N,
                                    int M);

// C = C - A * B
std::vector<double> gemm_grad(std::vector<double> A,
                              std::vector<double> B,
                              std::vector<double> R,
                              int N,
                              int M);

// }}} --------------------------------------- end of BLAS for optimization step

#endif  // end of ADAPTER_MKL_H
