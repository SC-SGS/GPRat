#pragma once

#include "gprat/tile_data.hpp"

#include <hpx/cache/statistics/local_full_statistics.hpp>
#include <hpx/modules/actions.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/modules/cache.hpp>
#include <hpx/modules/components.hpp>
#include <hpx/modules/components_base.hpp>
#include <hpx/modules/runtime_components.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/serialization/serialize_buffer.hpp>
#include <span>
#include <utility>

GPRAT_NS_BEGIN

void register_distributed_tile_counters();
void record_transmission_time(std::int64_t elapsed_ns);

void track_tile_server_allocation(std::size_t size);
void track_tile_server_deallocation(std::size_t size);

namespace detail
{
hpx::util::cache::statistics::local_full_statistics &get_global_statistics();

///////////////////////////////////////////////////////////////////////////
class global_full_statistics
{
  public:
    using update_on_exit = hpx::util::cache::statistics::local_full_statistics::update_on_exit;

    // ReSharper disable once CppNonExplicitConversionOperator
    operator hpx::util::cache::statistics::local_full_statistics &() const { return get_global_statistics(); }

    void got_hit() noexcept { get_global_statistics().got_hit(); }

    void got_miss() noexcept { get_global_statistics().got_miss(); }

    void got_insertion() noexcept { get_global_statistics().got_insertion(); }

    void got_eviction() noexcept { get_global_statistics().got_eviction(); }

    void clear() noexcept { get_global_statistics().clear(); }
};
}  // namespace detail

template <typename T>
class tile_cache
{
    friend struct tile_cache_counters;

  public:
    tile_cache() :
        cache_(16)
    { }

    bool try_get(const hpx::naming::gid_type &key, std::size_t generation, mutable_tile_data<T> &cached_data)
    {
        std::lock_guard g(mutex_);
        hpx::naming::gid_type unused;
        entry e;
        if (cache_.get_entry(key, unused, e))
        {
            if (e.generation == generation)
            {
                cached_data = e.data;
                return true;
            }
            // Erase the obsolete entry
            cache_.erase([&](const auto &p) { return p.first == key; });
        }
        return false;
    }

    void insert(const hpx::naming::gid_type &key, std::size_t generation, const mutable_tile_data<T> &data)
    {
        std::lock_guard g(mutex_);
        cache_.insert(key, entry{ data, generation });
    }

    void clear() { cache_.clear(); }

  private:
    struct entry
    {
        mutable_tile_data<T> data;
        std::size_t generation = 0;
    };

    hpx::mutex mutex_;
    hpx::util::cache::lru_cache<hpx::naming::gid_type, entry, detail::global_full_statistics> cache_;
};

namespace server
{

/**
 * Server component owning a single tile's data.
 *
 * @tparam T Element type of the tile. Usually some numeric type like double or float. This class currently only
 * requires T to be serializable by HPX.
 */
template <typename T>
struct tile_holder : hpx::components::locking_hook<hpx::components::component_base<tile_holder<T>>>
{
    tile_holder() { track_tile_server_allocation(0); }

    explicit tile_holder(const mutable_tile_data<double> &data) :
        data_(data)
    {
        track_tile_server_allocation(data.size());
    }

    ~tile_holder() { track_tile_server_deallocation(data_.size()); }

    [[nodiscard]] mutable_tile_data<double> get_data() const { return data_; }

    void set_data(const mutable_tile_data<double> &data) { data_ = data; }

    // Every member function that has to be invoked remotely needs to be
    // wrapped into a component action.
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(tile_holder, get_data)
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(tile_holder, set_data)

  private:
    mutable_tile_data<double> data_;
};

template <typename T>
struct tile_manager_shared_data
{
    struct tile_entry
    {
        tile_entry() :
            locality_id(hpx::naming::invalid_locality_id)
        { }

        tile_entry(hpx::id_type tile, std::uint32_t locality_id) :
            tile(std::move(tile)),
            locality_id(locality_id)
        { }

        hpx::id_type tile;
        std::uint32_t locality_id;
        std::shared_ptr<tile_holder<T>> local_data;

      private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive &ar, unsigned)
        {
            ar & tile & locality_id;
        }
    };

    std::vector<tile_entry> tiles;

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & tiles;
    }
};

template <typename T>
struct tile_manager : hpx::components::component_base<tile_manager<T>>
{
    tile_manager(tile_manager_shared_data<T> &&data) :
        data_(std::move(data))
    { }

    mutable_tile_data<T> get_tile_data(std::size_t tile_index, std::size_t generation)
    {
        const auto &target_tile = data_.tiles[tile_index];

        // Best is always to rely on local data
        if (target_tile.local_data)
        {
            return target_tile.local_data->get_data();
        }

        // Next, try the tile cache - maybe we have current data
        {
            mutable_tile_data<T> cached_data;
            if (cache_.try_get(target_tile.tile.get_gid(), generation, cached_data))
            {
                return cached_data;
            }
        }

        hpx::chrono::high_resolution_timer timer;
        auto data = hpx::async(typename tile_holder<T>::get_data_action{}, target_tile.tile).get();

        record_transmission_time(timer.elapsed_nanoseconds());
        cache_.insert(target_tile.tile.get_gid(), generation, data);

        return data;
    }

    hpx::future<mutable_tile_data<T>> get_tile_data_async(std::size_t tile_index, std::size_t generation)
    {
        const auto &target_tile = data_.tiles[tile_index];

        // Best is always to rely on local data
        if (target_tile.local_data)
        {
            return hpx::make_ready_future(target_tile.local_data->get_data());
        }

        // Next, try the tile cache - maybe we have current data
        {
            mutable_tile_data<T> cached_data;
            if (cache_.try_get(target_tile.tile.get_gid(), generation, cached_data))
            {
                return hpx::make_ready_future(cached_data);
            }
        }

        return hpx::async(typename tile_holder<T>::get_data_action{}, target_tile.tile)
            .then(
                [this, generation, gid = target_tile.tile.get_gid(), timer = hpx::chrono::high_resolution_timer()](
                    hpx::future<mutable_tile_data<T>> &&f)
                {
                    record_transmission_time(timer.elapsed_nanoseconds());
                    auto data = f.get();
                    cache_.insert(gid, generation, data);
                    return data;
                });
    }

    hpx::future<void>
    set_tile_data_async(std::size_t tile_index, std::size_t generation, const mutable_tile_data<T> &data)
    {
        const auto &target_tile = data_.tiles[tile_index];

        if (target_tile.local_data)
        {
            target_tile.local_data->set_data(data);
            return hpx::make_ready_future();
        }

        // We'd lose this tile after writing it, best to put it in the cache for now
        cache_.insert(target_tile.tile.get_gid(), generation, data);

        typename tile_holder<T>::set_data_action act;
        return hpx::async(act, target_tile.tile, data);
    }

  private:
    tile_manager_shared_data<T> data_;
    tile_cache<T> cache_;
};

}  // namespace server

// DECLARATION macros (use in a single header)

#define GPRAT_REGISTER_TILE_HOLDER_DECLARATION_IMPL(type, name)                                                        \
    HPX_REGISTER_ACTION_DECLARATION(type::get_data_action, HPX_PP_CAT(_tile_holder_get_data_action_, name))            \
    HPX_REGISTER_ACTION_DECLARATION(type::set_data_action, HPX_PP_CAT(_tile_holder_set_data_action_, name))            \
    /**/

#define GPRAT_REGISTER_TILED_DATASET_DECLARATION(type, name)                                                           \
    typedef ::GPRAT_NS::server::tile_holder<type> HPX_PP_CAT(_server_tile_holder_, HPX_PP_CAT(type, name));            \
    GPRAT_REGISTER_TILE_HOLDER_DECLARATION_IMPL(HPX_PP_CAT(_server_tile_holder_, HPX_PP_CAT(type, name)), name)

// REGISTRATION macros (use in a single .cpp file)

#define GPRAT_REGISTER_TILE_HOLDER_IMPL(type, name)                                                                    \
    HPX_REGISTER_ACTION(type::get_data_action, HPX_PP_CAT(_tile_holder_get_data_action_, name))                        \
    HPX_REGISTER_ACTION(type::set_data_action, HPX_PP_CAT(_tile_holder_set_data_action_, name))                        \
    typedef ::hpx::components::component<type> HPX_PP_CAT(_server_tile_holder_component_, name);                       \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(_server_tile_holder_component_, name))                                           \
    /**/

#define GPRAT_REGISTER_TILE_MANAGER_IMPL(type, name)                                                                   \
    typedef ::hpx::components::component<type> HPX_PP_CAT(_server_tile_manager_component_, name);                      \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(_server_tile_manager_component_, name))                                          \
    /**/

#define GPRAT_REGISTER_TILED_DATASET(type, name)                                                                       \
    typedef ::GPRAT_NS::server::tile_holder<type> HPX_PP_CAT(_server_tile_holder_, HPX_PP_CAT(type, name));            \
    GPRAT_REGISTER_TILE_HOLDER_IMPL(HPX_PP_CAT(_server_tile_holder_, HPX_PP_CAT(type, name)), name)                    \
    typedef ::GPRAT_NS::server::tile_manager<type> HPX_PP_CAT(_server_tile_manager_, HPX_PP_CAT(type, name));          \
    GPRAT_REGISTER_TILE_MANAGER_IMPL(HPX_PP_CAT(_server_tile_manager_, HPX_PP_CAT(type, name)), name)

template <typename T>
class tiled_dataset_accessor;

template <typename T>
class tile_handle
{
  public:
    tile_handle() = default;

    tile_handle(std::vector<hpx::id_type> managers, std::size_t tile_index, std::size_t generation) :
        managers_(std::move(managers)),
        tile_index_(tile_index),
        generation_(generation)
    { }

    operator mutable_tile_data<T>() const { return get(); }

    mutable_tile_data<T> get() const { return get_local_manager()->get_tile_data(tile_index_, generation_); }

    hpx::future<mutable_tile_data<T>> get_async() const
    {
        return get_local_manager()->get_tile_data_async(tile_index_, generation_);
    }

    hpx::future<tile_handle> set_async(const mutable_tile_data<T> &data) const
    {
        return get_local_manager()
            ->set_tile_data_async(tile_index_, generation_ + 1, data)
            .then(
                [self = *this](hpx::future<void> &&) mutable
                {
                    ++self.generation_;
                    return self;
                });
    }

  private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & managers_ & tile_index_ & generation_;
    }

    std::shared_ptr<server::tile_manager<T>> get_local_manager() const
    {
        const auto here = hpx::get_locality_id();
        for (const auto &id : managers_)
        {
            if (here == hpx::naming::get_locality_id_from_id(id))
            {
                return hpx::get_ptr<server::tile_manager<T>>(hpx::launch::sync, id);
            }
        }

        throw std::runtime_error("This locality is not known");
    }

    // TODO: It would be best if the caller could give us the right manager already,
    // but since the amount of localities is somewhat limited, this will do for now.
    std::vector<hpx::id_type> managers_;
    std::size_t tile_index_;
    std::size_t generation_;
};

template <typename T>
using tiled_dataset = std::vector<hpx::shared_future<tile_handle<T>>>;

template <typename T>
tiled_dataset<T>
create_tiled_dataset(std::span<const std::pair<hpx::id_type, std::size_t>> targets, std::size_t num_tiles)
{
    using data_type = server::tile_manager_shared_data<T>;

    // First, create the actual tile data holders
    std::vector<hpx::future<std::vector<hpx::id_type>>> holders;
    holders.reserve(targets.size());
    for (const auto &target : targets)
    {
        holders.emplace_back(hpx::components::bulk_create_async<server::tile_holder<T>>(target.first, target.second));
    }

    // Next we prepare our shared data for the manager components
    data_type manager_data;
    manager_data.tiles.resize(num_tiles);

    std::size_t l = 0;
    for (std::size_t i = 0; i < targets.size(); ++i)
    {
        const auto locality = hpx::naming::get_locality_id_from_id(targets[i].first);
        for (hpx::id_type &id : holders[i].get())
        {
            manager_data.tiles[l++] = data_type::tile_entry(std::move(id), locality);
            if (l == num_tiles)
            {
                break;
            }
        }
    }
    HPX_ASSERT(l == num_tiles);

    // Now we move on to the manager components
    std::vector<hpx::id_type> managers;
    managers.reserve(targets.size());
    for (const auto &target : targets)
    {
        managers.emplace_back(hpx::components::create<server::tile_manager<T>>(target.first, manager_data));
    }

    // Finally, we create our fat tile_handles
    tiled_dataset<T> tiles;
    tiles.reserve(num_tiles);
    for (std::size_t i = 0; i < num_tiles; ++i)
    {
        tiles.push_back(hpx::make_ready_future(tile_handle<T>{managers, i, 0}));
    }
    return tiles;
}

GPRAT_NS_END
