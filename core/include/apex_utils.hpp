#ifndef APEX_STEPS_H
#define APEX_STEPS_H

#if GPRAT_APEX_STEPS

#include <apex_api.hpp>
#include <hpx/future.hpp>

/// @brief Alias for obtaining the current high-resolution time point.
inline auto now = std::chrono::high_resolution_clock::now;

/// @brief Computes the duration in nanoseconds between the current time and a given start time.
inline double diff(const std::chrono::high_resolution_clock::time_point &start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now() - start_time).count();
}

/**
 * @brief Starts a timer for the next step.
 *
 * When GPRAT_APEX_STEPS=ON, this macro initializes a new apex timer newTimer for the current scope.
 *
 * @param newTimer Identifier of the new timer variable to be declared
 */
#define GPRAT_START_TIMER(newTimer) auto newTimer = now()

/**
 * @brief Measures the duration of the previous step.
 *
 * When GPRAT_APEX_STEPS=ON, this macro blocks execution until all provided HPX futures are ready and samples the
 * duration of APEX timer oldTimer with label oldLabel.
 *
 * @param oldTimer Identifier of the existing timer variable to be sampled
 * @param oldLabel String label associated with the measured duration
 * @param ...      Variadic arguments representing HPX futures to wait on
 */
#define GPRAT_STOP_TIMER(newTimer, oldTimer, oldLabel, ...)                                                            \
    hpx::wait_all(__VA_ARGS__);                                                                                        \
    apex::sample_value(oldLabel, diff(oldTimer))

#else

// Empty macro definitions when GPRAT_APEX_STEPS=OFF
#define GPRAT_START_TIMER(...)
#define GPRAT_STOP_TIMER(...)

#endif  // GPRAT_APEX_STEPS

#endif  // APEX_STEPS_H
