#pragma once

#include <hpx/cache/statistics/local_full_statistics.hpp>
#include <hpx/modules/actions.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/modules/cache.hpp>
#include <hpx/modules/components.hpp>
#include <hpx/modules/components_base.hpp>
#include <hpx/modules/runtime_components.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/serialization/serialize_buffer.hpp>
#include <span>

void register_distributed_tile_counters();
void record_transmission_time(std::int64_t elapsed_ns);

void track_tile_data_allocation(std::size_t size);
void track_tile_data_deallocation(std::size_t size);

void track_tile_server_allocation(std::size_t size);
void track_tile_server_deallocation(std::size_t size);

template <typename T>
struct tile_data
{
  private:
    typedef hpx::serialization::serialize_buffer<T> buffer_type;

    struct hold_reference
    {
        explicit hold_reference(const buffer_type &data) :
            data_(data)
        { }

        void operator()(const double *) const { }  // no deletion necessary

        buffer_type data_;
    };

    // In case we want pooling down the road...
    static T *allocate(std::size_t n)
    {
        track_tile_data_allocation(n);
        return new T[n];
    }

    static void deallocate(T *p) noexcept
    {
        track_tile_data_deallocation(0);  // we don't know here
        delete[] p;
    }

  public:
    tile_data() = default;

    // Create a new (uninitialized) partition of the given size.
    explicit tile_data(std::size_t size) :
        data_(allocate(size), size, buffer_type::take, &tile_data::deallocate)
    { }

    // Create a partition which acts as a proxy to a part of the embedded array.
    // The proxy is assumed to refer to either the left or the right boundary
    // element.
    tile_data(const tile_data &base, std::size_t offset, std::size_t size) :
        data_(base.data_.data() + offset,
              size,
              buffer_type::reference,
              hold_reference(base.data_))  // keep referenced partition alive
    { }

    [[nodiscard]] T *data() noexcept { return data_.data(); }

    [[nodiscard]] const T *data() const noexcept { return data_.data(); }

    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

    // ReSharper disable once CppNonExplicitConversionOperator
    operator std::span<T>() noexcept { return { data_.data(), data_.size() }; }  // NOLINT(*-explicit-constructor)

    // ReSharper disable once CppNonExplicitConversionOperator
    operator std::span<const T>() const noexcept  // NOLINT(*-explicit-constructor)
    {
        return { data_.data(), data_.size() };
    }

  private:
    // Serialization support: even if all of the code below runs on one
    // locality only, we need to provide an (empty) implementation for the
    // serialization as all arguments passed to actions have to support this.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        // clang-format off
        ar & data_;
        // clang-format on
    }

    buffer_type data_;
};

///////////////////////////////////////////////////////////////////////////////
// This is the server side representation of the data. We expose this as a HPX
// component which allows for it to be created and accessed remotely through
// a global address (hpx::id_type).
struct tile_server : hpx::components::component_base<tile_server>
{
    // construct new instances
    tile_server() = default;

    explicit tile_server(const tile_data<double> &data) :
        data_(data)
    {
        track_tile_server_allocation(data.size());
    }

    ~tile_server()
    {
        track_tile_server_deallocation(data_.size());
    }

    tile_data<double> get_data() const { return data_; }

    void set_data(const tile_data<double> &data) { data_ = data; }

    // Every member function that has to be invoked remotely needs to be
    // wrapped into a component action.
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(tile_server, get_data, get_data_action)
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(tile_server, set_data, set_data_action)

  private:
    tile_data<double> data_;
};

HPX_REGISTER_ACTION_DECLARATION(tile_server::get_data_action, get_data_action);
HPX_REGISTER_ACTION_DECLARATION(tile_server::set_data_action, set_data_action);

///////////////////////////////////////////////////////////////////////////////
// This is a client side helper class allowing to hide some of the tedious
// boilerplate while referencing a remote partition.

class tile_cache
{
    friend struct tile_cache_counters;

  public:
    tile_cache();

    bool try_get(const hpx::naming::gid_type &key, tile_data<double> &cached_data);

    void insert(const hpx::naming::gid_type &key, const tile_data<double> &data);

  private:
    hpx::mutex mutex_;
    hpx::util::cache::
        lru_cache<hpx::naming::gid_type, tile_data<double>, hpx::util::cache::statistics::local_full_statistics>
            cache_;
};

inline tile_cache &get_tile_cache()
{
    static tile_cache cache;
    return cache;
}

struct tile_handle : hpx::components::client_base<tile_handle, tile_server>
{
    using base_type = hpx::components::client_base<tile_handle, tile_server>;

    tile_handle() = default;

    // Create new component on locality 'where' and initialize the held data
    tile_handle(const hpx::id_type &where, const tile_data<double> &data) :
        base_type(hpx::new_<tile_server>(where, data))
    { }

    // Create new component on locality 'where' and initialize the held data
    template <typename T>
    requires hpx::traits::is_distribution_policy_v<T> tile_handle(const T &policy, const tile_data<double> &data) :
        base_type(hpx::new_<tile_server>(policy, data))
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
    // TODO: profile whether this strategy (new components on mutation) beats mutable tile state.
    [[nodiscard]] hpx::future<tile_data<double>> get_data() const
    {
        hpx::naming::gid_type current_gid = get().get_gid();
        {
            auto &cache = get_tile_cache();
            tile_data<double> cached_data;
            if (cache.try_get(current_gid, cached_data))
            {
                return hpx::make_ready_future(cached_data);
            }
        }

        tile_server::get_data_action act;
        return hpx::dataflow(
            [self = *this, current_gid, timer = hpx::chrono::high_resolution_timer()](
                hpx::future<tile_data<double>> f) mutable
            {
                (void) self;  // need this ref-counted object to stay alive!
                auto data = f.get();
                record_transmission_time(timer.elapsed_nanoseconds());
                get_tile_cache().insert(current_gid, data);
                return std::move(data);
            },
            hpx::async(act, get_id()));
    }

    [[nodiscard]] hpx::future<void> set_data(const tile_data<double> &data)
    {
        tile_server::set_data_action act;
        return hpx::async(act, get_id(), data);
    }
};
