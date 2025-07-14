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

void track_tile_data_allocation(std::size_t size);
void track_tile_data_deallocation(std::size_t size);

void track_tile_server_allocation(std::size_t size);
void track_tile_server_deallocation(std::size_t size);

namespace server
{
struct tiled_dataset_config_data
{
    struct tile_entry
    {
        tile_entry() :
            locality_id(hpx::naming::invalid_locality_id),
            generation(0)
        { }

        tile_entry(hpx::id_type tile, std::uint32_t locality_id, std::uint64_t generation) :
            tile(std::move(tile)),
            locality_id(locality_id),
            generation(generation)
        { }

        hpx::id_type tile;
        std::uint32_t locality_id;
        std::uint64_t generation;

      private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive &ar, unsigned)
        {
            ar & tile & locality_id & generation;
        }
    };

    tiled_dataset_config_data() = default;

    tiled_dataset_config_data(std::vector<tile_entry> &&tiles) :
        tiles(std::move(tiles))
    { }

    std::vector<tile_entry> tiles;

  private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & tiles;
    }
};
}  // namespace server

GPRAT_NS_END

HPX_DISTRIBUTED_METADATA_DECLARATION(GPRAT_NS::server::tiled_dataset_config_data,
                                     gprat_server_tiled_dataset_config_data)

GPRAT_NS_BEGIN

namespace server
{
// This is the server side representation of the data. We expose this as a HPX
// component which allows for it to be created and accessed remotely through
// a global address (hpx::id_type).
template <typename T>
struct tile_server : hpx::components::locking_hook<hpx::components::component_base<tile_server<T>>>
{
    tile_server() = default;

    explicit tile_server(const mutable_tile_data<double> &data) :
        data_(data)
    {
        track_tile_server_allocation(data.size());
    }

    ~tile_server() { track_tile_server_deallocation(data_.size()); }

    [[nodiscard]] mutable_tile_data<double> get_data() const { return data_; }

    void set_data(const mutable_tile_data<double> &data) { data_ = data; }

    // Every member function that has to be invoked remotely needs to be
    // wrapped into a component action.
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(tile_server, get_data)
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(tile_server, set_data)

  private:
    mutable_tile_data<double> data_;
};

}  // namespace server

#define GPRAT_REGISTER_TILED_DATASET_DECLARATION_IMPL(type, name)                                                      \
    HPX_REGISTER_ACTION_DECLARATION(type::get_data_action, HPX_PP_CAT(_tiled_dataset_get_data_action_, name))          \
    HPX_REGISTER_ACTION_DECLARATION(type::set_data_action, HPX_PP_CAT(_tiled_dataset_set_data_action_, name))          \
    /**/

#define GPRAT_REGISTER_TILED_DATASET_DECLARATION(type, name)                                                           \
    typedef ::GPRAT_NS::server::tile_server<type> HPX_PP_CAT(_tiled_dataset_server_, HPX_PP_CAT(type, name));          \
    GPRAT_REGISTER_TILED_DATASET_DECLARATION_IMPL(HPX_PP_CAT(_tiled_dataset_server_, HPX_PP_CAT(type, name)), name)

#define GPRAT_REGISTER_TILED_DATASET_IMPL(type, name)                                                                  \
    HPX_REGISTER_ACTION(type::get_data_action, HPX_PP_CAT(_tiled_dataset_get_data_action_, name))                      \
    HPX_REGISTER_ACTION(type::set_data_action, HPX_PP_CAT(_tiled_dataset_set_data_action_, name))                      \
    typedef ::hpx::components::component<type> HPX_PP_CAT(_tiled_dataset_server_component_, name);                     \
    HPX_REGISTER_COMPONENT(HPX_PP_CAT(_tiled_dataset_server_component_, name))                                         \
    /**/

#define GPRAT_REGISTER_TILED_DATASET(type, name)                                                                       \
    typedef ::GPRAT_NS::server::tile_server<type> HPX_PP_CAT(_tiled_dataset_server_, HPX_PP_CAT(type, name));          \
    GPRAT_REGISTER_TILED_DATASET_IMPL(HPX_PP_CAT(_tiled_dataset_server_, HPX_PP_CAT(type, name)), name)

template <typename T>
class tiled_dataset_accessor;

template <typename T>
class tile_handle
{
  public:
    tile_handle() = default;

    tile_handle(const hpx::id_type &id, std::size_t tile_index, std::size_t generation) :
        ds_(id),
        tile_index_(tile_index),
        generation_(generation)
    { }

    operator mutable_tile_data<T>() const { return get(); }

    mutable_tile_data<T> get() const
    {
        tiled_dataset_accessor<T> ds(ds_);  // TRANSITION
        return ds.get_tile_data(tile_index_, generation_).get();
    }

    hpx::future<mutable_tile_data<T>> get_async() const
    {
        tiled_dataset_accessor<T> ds(ds_);  // TRANSITION
        return ds.get_tile_data(tile_index_, generation_);
    }

    hpx::future<tile_handle> set_async(const mutable_tile_data<T> &data) const
    {
        tiled_dataset_accessor<T> ds(ds_);  // TRANSITION
        return ds.set_tile_data(tile_index_, generation_ + 1, data);
    }

  private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & ds_ & tile_index_ & generation_;
    }

    // we need this instead of the actual tile_handle because per-locality caches
    // reside in the accessor.
    hpx::id_type ds_;
    std::size_t tile_index_;
    std::size_t generation_;
};

template <typename T>
using tiled_dataset = std::vector<hpx::shared_future<tile_handle<T>>>;

template <typename T>
class tiled_dataset_accessor
    : public hpx::components::client_base<
          tiled_dataset_accessor<T>,
          hpx::components::server::distributed_metadata_base<server::tiled_dataset_config_data>>
{
    using server_type = hpx::components::server::distributed_metadata_base<server::tiled_dataset_config_data>;
    using base_type = hpx::components::client_base<tiled_dataset_accessor<T>, server_type>;

    using tile_server = server::tile_server<T>;

    struct tile_entry : server::tiled_dataset_config_data::tile_entry
    {
        using base_type = server::tiled_dataset_config_data::tile_entry;

        tile_entry() = default;

        tile_entry(const hpx::id_type &part, std::uint32_t locality_id, std::size_t version) :
            base_type(part, locality_id, version)
        { }

        tile_entry(const base_type &base) noexcept :
            base_type(base)
        { }

        tile_entry(base_type &&base) noexcept :
            base_type(HPX_MOVE(base))
        { }

        std::shared_ptr<tile_server> local_data;
    };

    // The list of partitions belonging to this vector.
    // Each partition is described by its corresponding client object, its
    // size, and locality id.
    using tiles_vector_type = std::vector<tile_entry>;

  public:
    explicit tiled_dataset_accessor(const hpx::id_type &id) { connect_to(id).get(); }

    explicit tiled_dataset_accessor(std::span<const std::pair<hpx::id_type, std::size_t>> targets,
                                    std::size_t num_tiles)
    {
        create(targets, num_tiles);
    }

    hpx::future<void> connect_to(const hpx::id_type &id)
    {
        return hpx::async(server_type::get_action(), id)
            .then([this, id](hpx::future<server::tiled_dataset_config_data> &&f) -> void
                  { return assign_existing(id, f.get()); });
    }

    hpx::future<mutable_tile_data<T>> get_tile_data(std::size_t tile_index, std::size_t /*generation*/) const
    {
        if (tiles_[tile_index].local_data)
        {
            return hpx::make_ready_future(tiles_[tile_index].local_data->get_data());
        }

        typename server::tile_server<T>::get_data_action act;
        return hpx::async(act, tiles_[tile_index].tile);
    }

    hpx::future<tile_handle<T>>
    set_tile_data(std::size_t tile_index, std::size_t generation, const mutable_tile_data<T> &data) const
    {
        if (tiles_[tile_index].local_data)
        {
            tiles_[tile_index].local_data->set_data(data);
            return hpx::make_ready_future(tile_handle<T>{ base_type::get(), tile_index, generation + 1 });
        }

        typename server::tile_server<T>::set_data_action act;
        return hpx::async(act, tiles_[tile_index].tile, data)
            .then([this, tile_index, generation](const hpx::future<void> &)
                  { return tile_handle<T>{ base_type::get(), tile_index, generation + 1 }; });
    }

    // TRANSITION
    tiled_dataset<T> to_dataset()
    {
        tiled_dataset<T> result;
        result.reserve(tiles_.size());
        for (std::size_t i = 0; i < tiles_.size(); ++i)
        {
            result.emplace_back(hpx::make_ready_future(tile_handle<T>{ base_type::get(), i, tiles_[i].generation }));
        }
        return result;
    }

  private:
    void assign_existing(const hpx::id_type &id, server::tiled_dataset_config_data &&config)
    {
        tiles_.clear();
        tiles_.insert(tiles_.end(), config.tiles.begin(), config.tiles.end());

        const auto here = hpx::get_locality_id();
        for (auto &tile : tiles_)
        {
            if (tile.locality_id == here && !tile.local_data)
            {
                tile.local_data = hpx::get_ptr<tile_server>(hpx::launch::sync, tile.tile);
            }
        }

        return base_type::reset(id);
    }

    void create(std::span<const std::pair<hpx::id_type, std::size_t>> targets, std::size_t num_tiles)
    {
        std::vector<hpx::future<std::vector<hpx::id_type>>> objs;
        objs.reserve(targets.size());
        for (const auto &target : targets)
        {
            objs.emplace_back(hpx::components::bulk_create_async<tile_server>(target.first, target.second));
        }

        const auto here = hpx::get_locality_id();
        tiles_.resize(num_tiles);

        std::size_t l = 0;
        for (std::size_t i = 0; i < targets.size(); ++i)
        {
            const auto locality = hpx::naming::get_locality_id_from_id(targets[i].first);
            for (const hpx::id_type &id : objs[i].get())
            {
                tiles_[l] = tile_entry(id, locality, 0);

                if (locality == here)
                {
                    tiles_[l].local_data = hpx::get_ptr<tile_server>(hpx::launch::sync, id);
                }

                if (++l == num_tiles)
                {
                    break;
                }
            }
        }
        HPX_ASSERT(l == num_tiles);

        std::vector<server::tiled_dataset_config_data::tile_entry> data{ tiles_.begin(), tiles_.end() };
        base_type::reset(
            hpx::new_<hpx::components::server::distributed_metadata_base<server::tiled_dataset_config_data>>(
                hpx::find_here(), server::tiled_dataset_config_data{ std::move(data) }));
    }

    tiles_vector_type tiles_;
};

// partitioned_vector_partition => tile_handle?
// server::partitioned_vector => tile_server?

/*template <typename T>
class tiled_dataset
{


  hpx::future<tile_reference<T>>& operator[](std::size_t index)
  {
    return
  }
  // operator[] => future[tile_reference]&
};*/

// OLD:
/*
///////////////////////////////////////////////////////////////////////////////
// This is a client side helper class allowing to hide some of the tedious
// boilerplate while referencing a remote partition.

template <typename T>
struct tile_handle : hpx::components::client_base<tile_handle<T>, server::tile_server<T>>
{
    using base_type = hpx::components::client_base<tile_handle, server::tile_server<T>>;

    tile_handle() = default;

    // Create new component on locality 'where' and initialize the held data
    tile_handle(const hpx::id_type &where, const mutable_tile_data<T> &data) :
        base_type(hpx::new_<server::tile_server<T>>(where, data))
    { }

    // Create new component on locality 'where' and initialize the held data
    template <typename T>
    requires hpx::traits::is_distribution_policy_v<T> tile_handle(const T &policy, const mutable_tile_data<T> &data) :
        base_type(hpx::new_<server::tile_server<T>>(policy, data))
    { }

    // Attach a future representing a (possibly remote) partition.
    // ReSharper disable once CppNonExplicitConvertingConstructor
    tile_handle(hpx::future<hpx::id_type> &&id) noexcept :
        base_type(std::move(id))
    { }

    // Unwrap a future<tile_handle> (a tile_handle already is a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    // ReSharper disable once CppNonExplicitConvertingConstructor
    tile_handle(hpx::future<tile_handle> &&c) noexcept :
        base_type(std::move(c))
    { }

    ///////////////////////////////////////////////////////////////////////////
    // tile's are immutable for now, meaning we can cache them on first use.
    [[nodiscard]] hpx::future<mutable_tile_data<double>> get_data() const
    {
        typename server::tile_server<T>::get_data_action act;
        return hpx::async(act, base_type::get_id());
    }

    [[nodiscard]] hpx::future<void> set_data(const mutable_tile_data<double> &data)
    {
        typename server::tile_server<T>::set_data_action act;
        return hpx::async(act, base_type::get_id(), data);
    }
};
*/

GPRAT_NS_END

/*
// serialization of partitioned_vector requires special handling
template <typename T>
struct hpx::traits::needs_reference_semantics<GPRAT_NS::tiled_dataset_accessor<T>>
  : std::true_type
{
};*/
