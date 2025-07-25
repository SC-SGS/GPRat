#pragma once

#include "distributed_tile.hpp"
#include "scheduling.hpp"
#include <hpx/runtime_distributed/find_localities.hpp>

GPRAT_NS_BEGIN

template <typename T>
std::vector<hpx::shared_future<mutable_tile_data<T>>>
make_cholesky_dataset(const tiled_scheduler_local &, std::size_t num_tiles)
{
    return { num_tiles * num_tiles };
}

// Default implementations in case the scheduler provides none
constexpr std::size_t cholesky_tile(...) { return 0; }

constexpr std::size_t cholesky_POTRF(...) { return 0; }

constexpr std::size_t cholesky_SYRK(...) { return 0; }

constexpr std::size_t cholesky_TRSM(...) { return 0; }

constexpr std::size_t cholesky_GEMM(...) { return 0; }

namespace scheduler
{

struct tiled_cholesky_scheduler_paap12 : tiled_scheduler_distributed
{
    using tiled_scheduler_distributed::tiled_scheduler_distributed;

    std::size_t num_localities = localities_.size();
};

template <typename T>
tiled_dataset<T> make_cholesky_dataset(const tiled_cholesky_scheduler_paap12 &sched, std::size_t num_tiles)
{
    std::vector<std::pair<hpx::id_type, std::size_t>> targets;
    targets.reserve(sched.num_localities);

    for (std::size_t i = 0; i < sched.num_localities; ++i)
    {
        targets.emplace_back(sched.localities_[i], 0);
    }

    for (std::size_t row = 0; row < num_tiles; row++)
    {
        for (std::size_t col = 0; col < num_tiles; col++)
        {
            const auto l = (row + col) % sched.num_localities;
            ++targets[l].second;
        }
    }

    return tiled_dataset_accessor<T>{ targets, num_tiles * num_tiles }.to_dataset();
}

constexpr std::size_t
cholesky_tile(const tiled_cholesky_scheduler_paap12 &sched, std::size_t /*n_tiles*/, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
cholesky_POTRF(const tiled_cholesky_scheduler_paap12 &sched, std::size_t /*n_tiles*/, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t
cholesky_SYRK(const tiled_cholesky_scheduler_paap12 &sched, std::size_t /*n_tiles*/, std::size_t m)
{
    return (2 * m) % sched.num_localities;
}

constexpr std::size_t
cholesky_TRSM(const tiled_cholesky_scheduler_paap12 &sched, std::size_t /*n_tiles*/, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t cholesky_GEMM(const tiled_cholesky_scheduler_paap12 &sched,
                                    std::size_t /*n_tiles*/,
                                    std::size_t /*k*/,
                                    std::size_t m,
                                    std::size_t n)
{
    return (m + n) % sched.num_localities;
}

// ==========================

struct tiled_cholesky_scheduler_cyclic : tiled_scheduler_distributed
{
    using tiled_scheduler_distributed::tiled_scheduler_distributed;

    std::size_t num_localities = localities_.size();
};

template <typename T>
tiled_dataset<T> make_cholesky_dataset(const tiled_cholesky_scheduler_cyclic &sched, std::size_t num_tiles)
{
    std::vector<std::pair<hpx::id_type, std::size_t>> targets;
    targets.reserve(sched.num_localities);

    for (std::size_t i = 0; i < sched.num_localities; ++i)
    {
        targets.emplace_back(sched.localities_[i], 0);
    }

    for (std::size_t row = 0; row < num_tiles; row++)
    {
        for (std::size_t col = 0; col < num_tiles; col++)
        {
            const auto l = (row * num_tiles + col) % sched.num_localities;
            ++targets[l].second;
        }
    }

    return tiled_dataset_accessor<T>{ targets, num_tiles * num_tiles }.to_dataset();
}

constexpr std::size_t
cholesky_tile(const tiled_cholesky_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row * n_tiles + col) % sched.num_localities;
}

constexpr std::size_t
cholesky_POTRF(const tiled_cholesky_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return (k * n_tiles + k) % sched.num_localities;
}

constexpr std::size_t
cholesky_SYRK(const tiled_cholesky_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t m)
{
    return (m * n_tiles + m) % sched.num_localities;
}

constexpr std::size_t
cholesky_TRSM(const tiled_cholesky_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k, std::size_t m)
{
    return (m * n_tiles + k) % sched.num_localities;
}

constexpr std::size_t cholesky_GEMM(const tiled_cholesky_scheduler_cyclic &sched,
                                    std::size_t n_tiles,
                                    std::size_t /*k*/,
                                    std::size_t m,
                                    std::size_t n)
{
    return (m * n_tiles + n) % sched.num_localities;
}

}  // namespace scheduler

GPRAT_NS_END
