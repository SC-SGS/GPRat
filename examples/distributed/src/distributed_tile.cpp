#include "distributed_tile.hpp"

namespace hpx::util
{

// This is explicitly instantiated to ensure that the id is stable
// across shared libraries.
template <>
extra_data_id_type extra_data_helper<tile_handle_cache>::id() noexcept
{
    static std::uint8_t id = 0;
    return &id;
}

template <>
void extra_data_helper<tile_handle_cache>::reset(tile_handle_cache *data) noexcept
{
    data->cached_data.reset();
}

}  // namespace hpx::util

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
