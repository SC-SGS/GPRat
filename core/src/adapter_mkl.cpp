#include "../include/adapter_mkl.hpp"

#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// BLAS operations for tiled cholkesy
// in-place Cholesky decomposition of A -> return factorized matrix L
hpx::shared_future<std::vector<double>> potrf(hpx::shared_future<std::vector<double>> f_A,
                              std::size_t N)
{
  auto A = f_A.get();
  // use ?potrf2 recursive version for better stability
  // POTRF - caution with dpotrf
  LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
  // return vector
  return hpx::make_ready_future(A);
}

// in-place solve X * L^T = A where L lower triangular
hpx::shared_future<std::vector<double>> trsm(hpx::shared_future<std::vector<double>> f_L,
                             hpx::shared_future<std::vector<double>> f_A,
                             std::size_t N)
{
  auto L = f_L.get();
  auto A = f_A.get();
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, N, N, alpha, L.data(), N, A.data(), N);
  // return vector
  return hpx::make_ready_future(A);
}

// A = A - B * B^T
hpx::shared_future<std::vector<double>> syrk(hpx::shared_future<std::vector<double>> f_A,
                             hpx::shared_future<std::vector<double>> f_B,
                             std::size_t N)
{
  auto B = f_B.get();
  auto A = f_A.get();
 // SYRK constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // SYRK kernel - caution with dsyrk
  cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
              N, N, alpha, B.data(), N, beta, A.data(), N);
  // return vector
  return hpx::make_ready_future(A);
}

// C = C - A * B^T
hpx::shared_future<std::vector<double>> gemm(hpx::shared_future<std::vector<double>> f_A,
                             hpx::shared_future<std::vector<double>> f_B,
                             hpx::shared_future<std::vector<double>> f_C,
                             std::size_t N)
{
  auto C = f_C.get();
  auto B = f_B.get();
  auto A = f_A.get();
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
  // return vector
  return hpx::make_ready_future(C);
}

// in-place solve L * x = a where L lower triangular
hpx::shared_future<std::vector<double>> trsv_l(hpx::shared_future<std::vector<double>> f_L,
                               hpx::shared_future<std::vector<double>> f_a,
                               std::size_t N)
{
  auto L = f_L.get();
  auto a = f_a.get();
  // TRSV kernel
  cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
              N, L.data(), N, a.data(), 1);
  // return vector
  return hpx::make_ready_future(a);
}

// b = b - A * a
hpx::shared_future<std::vector<double>> gemv_l(hpx::shared_future<std::vector<double>> f_A,
                               hpx::shared_future<std::vector<double>> f_a,
                               hpx::shared_future<std::vector<double>> f_b,
                               std::size_t N)
{
  auto A = f_A.get();
  auto a = f_a.get();
  auto b = f_b.get();
  // GEMV constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMV kernel
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, alpha,
              A.data(), N, a.data(), 1, beta, b.data(), 1);
  // return vector
  return hpx::make_ready_future(b);
}

// in-place solve L^T * x = a where L lower triangular
hpx::shared_future<std::vector<double>> trsv_u(hpx::shared_future<std::vector<double>> f_L,
                               hpx::shared_future<std::vector<double>> f_a,
                               std::size_t N)
{
  auto L = f_L.get();
  auto a = f_a.get();
  // TRSV kernel
  cblas_dtrsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit,
              N, L.data(), N, a.data(), 1);
  // return vector
  return hpx::make_ready_future(a);
}

// b = b - A^T * a
hpx::shared_future<std::vector<double>> gemv_u(hpx::shared_future<std::vector<double>> f_A,
                               hpx::shared_future<std::vector<double>> f_a,
                               hpx::shared_future<std::vector<double>> f_b,
                               std::size_t N)
{
  auto A = f_A.get();
  auto a = f_a.get();
  auto b = f_b.get();
  // GEMV constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMV kernel
  cblas_dgemv(CblasRowMajor, CblasTrans, N, N, alpha,
              A.data(), N, a.data(), 1, beta, b.data(), 1);
  // return vector
  return hpx::make_ready_future(b);
}

// A = y*beta^T + A
hpx::shared_future<std::vector<double>> ger(hpx::shared_future<std::vector<double>> f_A,
                            hpx::shared_future<std::vector<double>> f_x,
                            hpx::shared_future<std::vector<double>> f_y,
                            std::size_t N)
{
  auto A = f_A.get();
  auto x = f_x.get();
  auto y = f_y.get();
  // GER constants
  const double alpha = -1.0;
  // GER kernel
  cblas_dger(CblasRowMajor, N, N, alpha, x.data(), 1, y.data(), 1, A.data(), N);
  // return A
  return hpx::make_ready_future(A);
}

// C = C + A * B^T
hpx::shared_future<std::vector<double>> gemm_diag(hpx::shared_future<std::vector<double>> f_A,
                                  hpx::shared_future<std::vector<double>> f_B,
                                  hpx::shared_future<std::vector<double>> f_C,
                                  std::size_t N)
{
  auto C = f_C.get();
  auto B = f_B.get();
  auto A = f_A.get();
  // GEMM constants
  const double alpha = 1.0;
  const double beta = 1.0;
  // GEMM kernel
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
  // return vector
  return hpx::make_ready_future(C);
}

// BLAS operations for tiled prediction
// b = b + A * a where A(N_row, N_col), a(N_col) and b(N_row)
hpx::shared_future<std::vector<double>> gemv_p(hpx::shared_future<std::vector<double>> f_A,
                               hpx::shared_future<std::vector<double>> f_a,
                               hpx::shared_future<std::vector<double>> f_b,
                               std::size_t N_row,
                               std::size_t N_col)
{
  auto A = f_A.get();
  auto a = f_a.get();
  auto b = f_b.get();
  // GEMV constants
  const double alpha = 1.0;
  const double beta = 1.0;
  // GEMV kernel
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N_row, N_col, alpha,
              A.data(), N_col, a.data(), 1, beta, b.data(), 1);
  // return vector
  return hpx::make_ready_future(b);
}

////////////////////////////////////////////////////////////////////////////////
// BLAS operations used in uncertainty computation
// in-place solve X * L = A where L lower triangular
hpx::shared_future<std::vector<double>> trsm_l_KcK(hpx::shared_future<std::vector<double>> f_L,
                                   hpx::shared_future<std::vector<double>> f_A,
                                   std::size_t N,
                                   std::size_t M)
{
  auto L = f_L.get();
  auto A = f_A.get();  
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
  // return vector
  return hpx::make_ready_future(A);
}

// C = C - A * B
hpx::shared_future<std::vector<double>> gemm_l_KcK(hpx::shared_future<std::vector<double>> f_A,
                                   hpx::shared_future<std::vector<double>> f_B,
                                   hpx::shared_future<std::vector<double>> f_C,
                                   std::size_t N,
                                   std::size_t M)
{
  auto C = f_C.get();
  auto B = f_B.get();
  auto A = f_A.get();
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return hpx::make_ready_future(C);
}

// C = C - A^T * B
hpx::shared_future<std::vector<double>> gemm_cross_tcross_matrix(hpx::shared_future<std::vector<double>> f_A,
                                                 hpx::shared_future<std::vector<double>> f_B,
                                                 hpx::shared_future<std::vector<double>> f_C,
                                                 std::size_t N,
                                                 std::size_t M)
{
  auto C = f_C.get();
  auto B = f_B.get();
  auto A = f_A.get();
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              M, M, N, alpha, A.data(), M, B.data(), M, beta, C.data(), M);
  // return vector
  return hpx::make_ready_future(C);
}

////////////////////////////////////////////////////////////////////////////////
// BLAS operations used in optimization step
// in-place solve L * X = A where L lower triangular
hpx::shared_future<std::vector<double>> trsm_l_matrix(hpx::shared_future<std::vector<double>> f_L,
                                      hpx::shared_future<std::vector<double>> f_A,
                                      std::size_t N,
                                      std::size_t M)
{
  auto L = f_L.get();
  auto A = f_A.get();
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
  // return vector
  return hpx::make_ready_future(A);
}

// C = C - A * B
hpx::shared_future<std::vector<double>> gemm_l_matrix(hpx::shared_future<std::vector<double>> f_A,
                                      hpx::shared_future<std::vector<double>> f_B,
                                      hpx::shared_future<std::vector<double>> f_C,
                                      std::size_t N,
                                      std::size_t M)
{
  auto C = f_C.get();
  auto B = f_B.get();
  auto A = f_A.get(); 
  // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return hpx::make_ready_future(C);
}

// in-place solve L^T * X = A where L upper triangular
hpx::shared_future<std::vector<double>> trsm_u_matrix(hpx::shared_future<std::vector<double>> f_L,
                                      hpx::shared_future<std::vector<double>> f_A,
                                      std::size_t N,
                                      std::size_t M)
{
  auto L = f_L.get();
  auto A = f_A.get();
  // TRSM constants
  const double alpha = 1.0;
  // TRSM kernel - caution with dtrsm
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, N, M, alpha, L.data(), N, A.data(), M);
  // return vector
  return hpx::make_ready_future(A);
}

// C = C - A^T * B
{
  auto C = f_C.get();
  auto B = f_B.get();
  auto A = f_A.get();
 // GEMM constants
  const double alpha = -1.0;
  const double beta = 1.0;
  // GEMM kernel - caution with dgemm
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              N, M, N, alpha, A.data(), N, B.data(), M, beta, C.data(), M);
  // return vector
  return hpx::make_ready_future(C);
}

// Dot product used in dot calculation
double dot(std::size_t N,
          std::vector<double> a,
          std::vector<double> b)
{
  return cblas_ddot(N, a.data(), 1, b.data(), 1);
}

// C = C - A * B
hpx::shared_future<std::vector<double>> dot_uncertainty(hpx::shared_future<std::vector<double>> f_A,
                                        hpx::shared_future<std::vector<double>> f_R,
                                        std::size_t N,
                                        std::size_t M)
{
  auto R = f_R.get();
  auto A = f_A.get();
  for (int j = 0; j < M; ++j)
  {
    // Extract the j-th column and compute its dot product with itself
    R[j] += cblas_ddot(N, &A[j], M, &A[j], M);
  }
  return hpx::make_ready_future(R);
}

// C = C - A * B
hpx::shared_future<std::vector<double>> gemm_grad(hpx::shared_future<std::vector<double>> f_A,
                                  hpx::shared_future<std::vector<double>> f_B,
                                  hpx::shared_future<std::vector<double>> f_R,
                                  std::size_t N,
                                  std::size_t M)
{
  auto R = f_R.get();
  auto B = f_B.get();
  auto A = f_A.get();  
  for (std::size_t i = 0; i < N; ++i)
  {
    R[i] += cblas_ddot(M, &A[i * M], 1, &B[i], N);
  }
  return hpx::make_ready_future(R);
}
