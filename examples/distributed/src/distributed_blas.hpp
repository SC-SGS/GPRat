#pragma once

#include "gprat/cpu/adapter_cblas_fp64.hpp"

#include "distributed_tile.hpp"
#include "scheduling.hpp"
#include <hpx/actions_base/plain_action.hpp>

GPRAT_REGISTER_TILED_DATASET_DECLARATION(double, double);

GPRAT_NS_BEGIN

hpx::future<tile_handle<double>> potrf_distributed(const tile_handle<double> &A, int N);
hpx::future<tile_handle<double>> trsm_distributed(
    const tile_handle<double> &L,
    const tile_handle<double> &A,
    int N,
    int M,
    BLAS_TRANSPOSE transpose_L,
    BLAS_SIDE side_L);
hpx::future<tile_handle<double>> syrk_distributed(const tile_handle<double> &A, const tile_handle<double> &B, int N);
hpx::future<tile_handle<double>> gemm_distributed(
    const tile_handle<double> &A,
    const tile_handle<double> &B,
    const tile_handle<double> &C,
    int N,
    int M,
    int K,
    BLAS_TRANSPOSE transpose_A,
    BLAS_TRANSPOSE transpose_B);

hpx::future<tile_handle<double>>
trsv_distributed(const tile_handle<double> &L, const tile_handle<double> &a, int N, BLAS_TRANSPOSE transpose_L);
hpx::future<tile_handle<double>> gemv_distributed(
    const tile_handle<double> &A,
    const tile_handle<double> &a,
    const tile_handle<double> &b,
    int N,
    int M,
    BLAS_ALPHA alpha,
    BLAS_TRANSPOSE transpose_A);

hpx::future<tile_handle<double>>
dot_diag_syrk_distributed(const tile_handle<double> &A, const tile_handle<double> &r, int N, int M);
hpx::future<tile_handle<double>> dot_diag_gemm_distributed(
const tile_handle<double> &A, const tile_handle<double> &B, const tile_handle<double> &r, int N, int M);
hpx::future<tile_handle<double>> axpy_distributed(
    const tile_handle<double> &y, const tile_handle<double> &x, int N);

HPX_DEFINE_PLAIN_DIRECT_ACTION(potrf_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(trsm_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(syrk_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gemm_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(trsv_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gemv_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(dot_diag_syrk_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(dot_diag_gemm_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(axpy_distributed);

GPRAT_DECLARE_PLAIN_ACTION_FOR(&potrf, potrf_distributed_action, "POTRF");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&trsm, trsm_distributed_action, "TRSM");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&syrk, syrk_distributed_action, "SYRK");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&gemm, gemm_distributed_action, "GEMM");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&trsv, trsv_distributed_action, "TRSV");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&gemv, gemv_distributed_action, "GEMV");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&dot_diag_syrk, dot_diag_syrk_distributed_action, "dot diag(SYRK)");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&dot_diag_gemm, dot_diag_gemm_distributed_action, "dot diag(GEMM)");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&axpy, axpy_distributed_action, "axpy");

GPRAT_NS_END

HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::potrf_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::trsm_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::syrk_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::gemm_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::trsv_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::gemv_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::dot_diag_syrk_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::dot_diag_gemm_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::axpy_distributed_action);
