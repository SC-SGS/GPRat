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

HPX_DEFINE_PLAIN_DIRECT_ACTION(potrf_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(trsm_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(syrk_distributed);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gemm_distributed);

GPRAT_DECLARE_PLAIN_ACTION_FOR(&potrf, potrf_distributed_action, "POTRF");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&trsm, trsm_distributed_action, "TRSM");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&syrk, syrk_distributed_action, "SYRK");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&gemm, gemm_distributed_action, "GEMM");

void register_distributed_blas_counters();

GPRAT_NS_END

HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::potrf_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::trsm_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::syrk_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(GPRAT_NS::gemm_distributed_action);
