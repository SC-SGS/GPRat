#include "distributed_tile.hpp"

#include <atomic>
#include <hpx/include/performance_counters.hpp>

GPRAT_NS_BEGIN

namespace detail
{
hpx::util::cache::statistics::local_full_statistics &get_global_statistics()
{
    static hpx::util::cache::statistics::local_full_statistics stats;
    return stats;
}

}  // namespace detail

std::atomic<std::uint64_t> tile_transmission_time(0);
std::atomic<std::uint64_t> tile_transmission_count(0);
std::atomic<std::uint64_t> tile_data_allocations(0);
std::atomic<std::uint64_t> tile_data_deallocations(0);
std::atomic<std::uint64_t> tile_server_allocations(0);
std::atomic<std::uint64_t> tile_server_deallocations(0);

void record_transmission_time(std::int64_t elapsed_ns)
{
    HPX_ASSERT(elapsed_ns >= 0);
    tile_transmission_count += 1;
    if (elapsed_ns > 0)
    {
        tile_transmission_time += static_cast<std::uint64_t>(elapsed_ns);
    }
}

void track_tile_server_allocation(std::size_t /*size*/) { tile_server_allocations += 1; }

void track_tile_server_deallocation(std::size_t /*size*/) { tile_server_deallocations += 1; }

#define GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(name)                                                                       \
    std::uint64_t get_##name(bool reset) { return hpx::util::get_and_reset_value(name, reset); }

GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_transmission_count)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_transmission_time)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_server_allocations)
GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(tile_server_deallocations)

#undef GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR

void register_distributed_tile_counters()
{
    // XXX: you can do this with templates, but it's quite a bit more complicated
#define GPRAT_MAKE_STATISTICS_ACCESSOR(name, stats_expr)                                                               \
    hpx::performance_counters::install_counter_type(                                                                   \
        name,                                                                                                          \
        [](bool reset) { return (stats_expr) (reset); },                                                               \
        #stats_expr,                                                                                                   \
        "",                                                                                                            \
        hpx::performance_counters::counter_type::monotonically_increasing)

    GPRAT_MAKE_STATISTICS_ACCESSOR("/gprat/tile_cache/hits", detail::get_global_statistics().hits);
    GPRAT_MAKE_STATISTICS_ACCESSOR("/gprat/tile_cache/misses", detail::get_global_statistics().misses);
    GPRAT_MAKE_STATISTICS_ACCESSOR("/gprat/tile_cache/evictions", detail::get_global_statistics().evictions);
    GPRAT_MAKE_STATISTICS_ACCESSOR("/gprat/tile_cache/insertions", detail::get_global_statistics().insertions);

#undef GPRAT_MAKE_STATISTICS_ACCESSOR

    hpx::performance_counters::install_counter_type(
        "/gprat/tile_cache/transmission_count",
        &get_tile_transmission_time,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);

    hpx::performance_counters::install_counter_type(
        "/gprat/tile_cache/transmission_time",
        &get_tile_transmission_time,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);

    hpx::performance_counters::install_counter_type(
        "/gprat/tile_server/num_allocations",
        &get_tile_server_allocations,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/tile_server/num_deallocations",
        &get_tile_server_deallocations,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
}

GPRAT_NS_END
