#pragma once

#include <hpx/async_distributed/dataflow.hpp>
#include <hpx/chrono.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/util/get_and_reset_value.hpp>

template <auto F>
struct plain_action_for;

#define GPRAT_DECLARE_PLAIN_ACTION_FOR(local_function, action, friendly_name)                                          \
    template <>                                                                                                        \
    struct plain_action_for<local_function>                                                                            \
    {                                                                                                                  \
        using action_type = action;                                                                                    \
        constexpr static std::string_view name = friendly_name;                                                        \
        static std::atomic<std::uint64_t> elapsed_ns_in_action;                                                        \
    }

#define GPRAT_DEFINE_PLAIN_ACTION_FOR(local_function)                                                                  \
    std::atomic<std::uint64_t> plain_action_for<local_function>::elapsed_ns_in_action(0)

template <auto F>
std::uint64_t get_and_reset_plain_action_elapsed(bool reset)
{
    return hpx::util::get_and_reset_value(plain_action_for<F>::elapsed_ns_in_action, reset);
}

// This is a simple tag-type like construct that exists solely, so we automatically pick the right dataflow() overload.
struct schedule_on_locality
{
    // conversion is intended here, we don't want people to actually spell this type out
    // ReSharper disable once CppNonExplicitConvertingConstructor
    schedule_on_locality(const hpx::id_type &where) :
        where(where)
    { }

    hpx::id_type where;
};

template <auto F, typename... Args>
decltype(auto) dataflow(const schedule_on_locality &on, Args &&...args)
{
    typename plain_action_for<F>::action_type act;
    return hpx::dataflow(
        [timer = hpx::chrono::high_resolution_timer()](auto &&r)
        {
            const auto elapsed = timer.elapsed_nanoseconds();
            HPX_ASSERT(elapsed >= 0);
            plain_action_for<F>::elapsed_ns_in_action += elapsed;
            return r.get();
        },
        hpx::dataflow(act, on.where, args...));
}
