#include "distributed_blas.hpp"

#include "gprat/cpu/adapter_cblas_fp64.hpp"

#include <hpx/distribution_policies/colocating_distribution_policy.hpp>
#include <hpx/include/performance_counters.hpp>

HPX_REGISTER_ACTION(GPRAT_NS::potrf_distributed_action);
HPX_REGISTER_ACTION(GPRAT_NS::trsm_distributed_action);
HPX_REGISTER_ACTION(GPRAT_NS::syrk_distributed_action);
HPX_REGISTER_ACTION(GPRAT_NS::gemm_distributed_action);

GPRAT_NS_BEGIN

GPRAT_DEFINE_PLAIN_ACTION_FOR(&potrf);
GPRAT_DEFINE_PLAIN_ACTION_FOR(&trsm);
GPRAT_DEFINE_PLAIN_ACTION_FOR(&syrk);
GPRAT_DEFINE_PLAIN_ACTION_FOR(&gemm);

hpx::future<tile_handle<double>> potrf_distributed(const tile_handle<double> &A, int N)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::unwrapping(
            [A, N](const mutable_tile_data<double> &tile)
            {
                GPRAT_TIME_PLAIN_ACTION(potrf);
                return A.set_async(potrf(tile, N));
            }),
        A.get_async());
}

hpx::future<tile_handle<double>> trsm_distributed(
    const tile_handle<double> &L,
    const tile_handle<double> &A,
    int N,
    int M,
    BLAS_TRANSPOSE transpose_L,
    BLAS_SIDE side_L)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::unwrapping(
            [A, N, M, transpose_L, side_L](const mutable_tile_data<double> &Ld, mutable_tile_data<double> Ad)
            {
                GPRAT_TIME_PLAIN_ACTION(trsm);
                return A.set_async(trsm(Ld, Ad, N, M, transpose_L, side_L));
            }),
        L.get_async(),
        A.get_async());
}

hpx::future<tile_handle<double>> syrk_distributed(const tile_handle<double> &A, const tile_handle<double> &B, int N)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::unwrapping(
            [A, N](mutable_tile_data<double> Ad, const mutable_tile_data<double> &Bd)
            {
                GPRAT_TIME_PLAIN_ACTION(syrk);
                return A.set_async(syrk(Ad, Bd, N));
            }),
        A.get_async(),
        B.get_async());
}

hpx::future<tile_handle<double>> gemm_distributed(
    const tile_handle<double> &A,
    const tile_handle<double> &B,
    const tile_handle<double> &C,
    int N,
    int M,
    int K,
    BLAS_TRANSPOSE transpose_A,
    BLAS_TRANSPOSE transpose_B)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::unwrapping(
            [C, N, M, K, transpose_A, transpose_B](
                const mutable_tile_data<double> &Ad, const mutable_tile_data<double> &Bd, mutable_tile_data<double> Cd)
            {
                GPRAT_TIME_PLAIN_ACTION(gemm);
                return C.set_async(gemm(Ad, Bd, Cd, N, M, K, transpose_A, transpose_B));
            }),
        A.get_async(),
        B.get_async(),
        C.get_async());
}

void register_distributed_blas_counters()
{
    hpx::performance_counters::install_counter_type(
        "/gprat/potrf/time",
        get_and_reset_plain_action_elapsed<&potrf>,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/trsm/time",
        get_and_reset_plain_action_elapsed<&trsm>,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/syrk/time",
        get_and_reset_plain_action_elapsed<&syrk>,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/gemm/time",
        get_and_reset_plain_action_elapsed<&gemm>,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
}

GPRAT_NS_END
