#include "distributed_tile.hpp"

#include <atomic>
#include <hpx/include/performance_counters.hpp>

tile_cache::tile_cache() :
    cache_(16)
{ }

bool tile_cache::try_get(const hpx::naming::gid_type &key, tile_data<double> &cached_data)
{
    std::lock_guard g(mutex_);
    hpx::naming::gid_type unused;
    return cache_.get_entry(key, unused, cached_data);
}

void tile_cache::insert(const hpx::naming::gid_type &key, const tile_data<double> &data)
{
    std::lock_guard g(mutex_);
    cache_.insert(key, data);
}

struct tile_cache_counters
{
    // XXX: you can do this with templates, but it's quite a bit more complicated
#define GPRAT_MAKE_STATISTICS_ACCESSOR(name)                                                                           \
    static std::uint64_t get_cache_##name(bool reset)                                                                  \
    {                                                                                                                  \
        auto &cache = get_tile_cache();                                                                                \
        std::lock_guard lock(cache.mutex_);                                                                            \
        return cache.cache_.get_statistics().name(reset);                                                              \
    }

    GPRAT_MAKE_STATISTICS_ACCESSOR(hits);
    GPRAT_MAKE_STATISTICS_ACCESSOR(misses);
    GPRAT_MAKE_STATISTICS_ACCESSOR(evictions);
    GPRAT_MAKE_STATISTICS_ACCESSOR(insertions);

#undef GPRAT_MAKE_STATISTICS_ACCESSOR
};

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
