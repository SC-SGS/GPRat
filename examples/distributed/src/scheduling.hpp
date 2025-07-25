#pragma once

#include "gprat/detail/config.hpp"

#include <hpx/async_distributed/dataflow.hpp>
#include <hpx/chrono.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/util/get_and_reset_value.hpp>

GPRAT_NS_BEGIN

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

struct plain_action_timer
{
    explicit plain_action_timer(std::atomic<std::uint64_t> &total) :
        total(total)
    { }

    ~plain_action_timer()
    {
        const auto elapsed = timer.elapsed_nanoseconds();
        HPX_ASSERT(elapsed >= 0);
        if (elapsed > 0)
            total += static_cast<std::uint64_t>(elapsed);
    }

    std::atomic<std::uint64_t> &total;
    hpx::chrono::high_resolution_timer timer;
};

#define GPRAT_TIME_PLAIN_ACTION(local_function)                                                                        \
    plain_action_timer _action_timer(plain_action_for<local_function>::elapsed_ns_in_action);

template <auto F>
std::uint64_t get_and_reset_plain_action_elapsed(bool reset)
{
    return hpx::util::get_and_reset_value(plain_action_for<F>::elapsed_ns_in_action, reset);
}

struct tiled_scheduler_distributed
{
    tiled_scheduler_distributed() :
        localities_(hpx::find_all_localities())
    {
        // ctor
    }

    explicit tiled_scheduler_distributed(std::vector<hpx::id_type> in_localities) :
        localities_(std::move(in_localities))
    {
        // ctor
    }

    std::vector<hpx::id_type> localities_;
};

struct tiled_scheduler_local
{ };

namespace detail
{
// HPX does not auto-collapse future chains in their async(), dataflow(), ... functions.
// This usually works fine, but we require shared_futures most of the time
// and the language will not do two-step conversions for us (future<future<R>> -> future<R> -> shared_future<R>).
// see: https://github.com/STEllAR-GROUP/hpx/issues/3758
template <typename R>
hpx::future<R> collapse(hpx::future<hpx::future<R>> &&fut)
{
    return { std::move(fut) };
}

template <typename R>
hpx::future<R> collapse(hpx::future<R> &&fut)
{
    return std::move(fut);
}

template <auto F, typename... Args>
decltype(auto)
named_dataflow(const tiled_scheduler_distributed &sched, std::size_t on, const char * /*name*/, Args &&...args)
{
    return collapse(hpx::dataflow(hpx::launch::async,
                                  hpx::unwrapping(typename plain_action_for<F>::action_type{}),
                                  sched.localities_[on],
                                  std::forward<Args>(args)...));
    /*return hpx::dataflow(
        [timer = hpx::chrono::high_resolution_timer()](auto &&r)
        {
            const auto elapsed = timer.elapsed_nanoseconds();
            HPX_ASSERT(elapsed >= 0);
            plain_action_for<F>::elapsed_ns_in_action += elapsed;
            return collapse(r.get());
        },
        hpx::dataflow(typename plain_action_for<F>::action_type{}, sched.localities_[on],
       std::forward<Args>(args)...));*/
}

template <auto F, typename... Args>
decltype(auto) named_dataflow(const tiled_scheduler_local &sched, std::size_t /*on*/, const char *name, Args &&...args)
{
    return hpx::dataflow(hpx::annotated_function(hpx::unwrapping(F), name), std::forward<Args>(args)...);
}

template <auto F, typename... Args>
decltype(auto) named_async(const tiled_scheduler_local &sched, std::size_t /*on*/, const char *name, Args &&...args)
{
    return hpx::async(hpx::annotated_function(F, name), std::forward<Args>(args)...);
}

}  // namespace detail

GPRAT_NS_END
