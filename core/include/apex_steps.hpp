#ifndef APEX_STEPS_H
#define APEX_STEPS_H

#if GPRAT_APEX_STEPS

#include <hpx/future.hpp>

/**
 * @brief Measures the duration of the previous step and starts a timer for the next.
 *
 * When GPRAT_APEX_STEPS=ON, this macro performs three actions:
 * 1. Blocks execution until all provided HPX futures are ready
 * 2. Samples the duration of APEX timer oldTimer with label oldLabel
 * 3. Initializes a new timer newTimer for the current scope
 *
 * @param newTimer Identifier of the new timer variable to be declared
 * @param oldTimer Identifier of the existing timer variable to be sampled
 * @param oldLabel String label associated with the measured duration
 * @param ...      Variadic arguments representing HPX futures to wait on
 */
#define GPRAT_MEASURE_AND_START_STEP(newTimer, oldTimer, oldLabel, ...)                                                \
    hpx::wait_all(__VA_ARGS__);                                                                                        \
    apex::sample_value(oldLabel, diff(oldTimer));                                                                         \
    auto newTimer = now()

#else

/**
 * @brief Empty macro that can be activated by compiling with GPRAT_APEX_STEPS=ON.
 *
 * If GPRAT_APEX_STEPS=ON, this macro is used for waiting until all provided futures are ready, then measuring
 * the duration of an old timer and starting a new timer.
 *
 * This macro does not require any arguments. For reference, the following parameters would be used when
 * GPRAT_APEX_STEPS=ON:
 * @param newTimer Identifier of the new timer to be created.
 * @param oldTimer Identifier of the old timer to be sampled.
 * @param oldLabel Label associated with the old timer.
 * @param ... Variadic arguments representing HPX futures to wait on.
 */
#define GPRAT_MEASURE_AND_START_STEP(...)

#endif  // GPRAT_APEX_STEPS

#endif  // APEX_STEPS_H
