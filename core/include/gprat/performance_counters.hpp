﻿#ifndef GPRAT_PERFORMANCE_COUNTERS_HPP
#define GPRAT_PERFORMANCE_COUNTERS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <hpx/modules/assertion.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <hpx/util/get_and_reset_value.hpp>

GPRAT_NS_BEGIN

/// The following is a very simple way of defining per-function metrics by using the function itself as a template
/// parameter ensuring that each function receives exactly one instantiation.
template <auto F>
struct function_performance_metrics
{
    /// Number of times the function was called
    static std::atomic<std::uint64_t> num_calls;

    /// Total wall-clock time elapsed inside the function
    static std::atomic<std::uint64_t> elapsed_ns;
};

template <auto F>
/*static*/ std::atomic<std::uint64_t> function_performance_metrics<F>::num_calls(0);
template <auto F>
/*static*/ std::atomic<std::uint64_t> function_performance_metrics<F>::elapsed_ns(0);

/// @brief This RAII helper allows us to time a function's total wall-clock execution time with minimal code.
struct scoped_function_timer
{
    explicit scoped_function_timer(std::atomic<std::uint64_t> &num_calls, std::atomic<std::uint64_t> &in_total) :
        total(in_total)
    {
        ++num_calls;
    }

    ~scoped_function_timer()
    {
        const auto elapsed = timer.elapsed_nanoseconds();
        HPX_ASSERT(elapsed >= 0);
        if (elapsed > 0)
        {
            total += static_cast<std::uint64_t>(elapsed);
        }
    }

    std::atomic<std::uint64_t> &total;
    hpx::chrono::high_resolution_timer timer;
};

/// @brief Time the execution of the enclosing function from the current point to its end.
/// @param local_function The function key that we're collecting performance information for. Usually the enclosing
/// function.
#define GPRAT_TIME_FUNCTION(local_function)                                                                            \
    scoped_function_timer _gprat_fn_timer(function_performance_metrics<local_function>::num_calls,                     \
                                          function_performance_metrics<local_function>::elapsed_ns)

template <auto F>
std::uint64_t get_and_reset_function_elapsed(bool reset)
{
    return hpx::util::get_and_reset_value(function_performance_metrics<F>::elapsed_ns, reset);
}

void track_tile_data_allocation(std::size_t size);
void track_tile_data_deallocation(std::size_t size);

void register_performance_counters();

GPRAT_NS_END

#endif
