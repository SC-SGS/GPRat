#pragma once

#include "cpu/adapter_cblas_fp64.hpp"
#include "distributed_tile.hpp"
#include "scheduling.hpp"
#include <hpx/actions_base/plain_action.hpp>

tile_handle potrf_distributed(const tile_handle &A, int N);
tile_handle trsm_distributed(
    const tile_handle &L, const tile_handle &A, int N, int M, BLAS_TRANSPOSE transpose_L, BLAS_SIDE side_L);
tile_handle syrk_distributed(const tile_handle &A, const tile_handle &B, int N);
tile_handle gemm_distributed(
    const tile_handle &A,
    const tile_handle &B,
    const tile_handle &C,
    int N,
    int M,
    int K,
    BLAS_TRANSPOSE transpose_A,
    BLAS_TRANSPOSE transpose_B);

HPX_DEFINE_PLAIN_ACTION(potrf_distributed);
HPX_DEFINE_PLAIN_ACTION(trsm_distributed);
HPX_DEFINE_PLAIN_ACTION(syrk_distributed);
HPX_DEFINE_PLAIN_ACTION(gemm_distributed);

GPRAT_DECLARE_PLAIN_ACTION_FOR(&inplace::potrf, potrf_distributed_action, "POTRF");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&inplace::trsm, trsm_distributed_action, "TRSM");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&inplace::syrk, syrk_distributed_action, "SYRK");
GPRAT_DECLARE_PLAIN_ACTION_FOR(&inplace::gemm, gemm_distributed_action, "GEMM");

void register_distributed_blas_counters();
