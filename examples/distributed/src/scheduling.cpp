#include "scheduling.hpp"

#include <atomic>
#include <hpx/include/performance_counters.hpp>

std::atomic<std::uint64_t> tile_transmission_time(0);

void record_transmission_time(std::int64_t elapsed_ns)
{
    HPX_ASSERT(elapsed_ns >= 0);
    tile_transmission_time += elapsed_ns;
}

std::uint64_t get_transmission_time(bool reset)
{
    return hpx::util::get_and_reset_value(tile_transmission_time, reset);
}

void register_distributed_tile_counters()
{
    hpx::performance_counters::install_counter_type(
        "/gprat/tile_cache/hits",
        &tile_cache_counters::get_cache_hits,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/tile_cache/misses",
        &tile_cache_counters::get_cache_misses,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/tile_cache/evictions",
        &tile_cache_counters::get_cache_evictions,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/tile_cache/insertions",
        &tile_cache_counters::get_cache_insertions,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
    hpx::performance_counters::install_counter_type(
        "/gprat/tile_cache/transmission_time",
        &get_transmission_time,
        "",
        "",
        hpx::performance_counters::counter_type::monotonically_increasing);
}

// The macros below are necessary to generate the code required for exposing
// our partition type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<tile_server> tile_server_type;
HPX_REGISTER_COMPONENT(tile_server_type, tile_server)

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef tile_server::get_data_action get_data_action;
HPX_REGISTER_ACTION(get_data_action)

typedef tile_server::set_data_action set_data_action;
HPX_REGISTER_ACTION(set_data_action)
