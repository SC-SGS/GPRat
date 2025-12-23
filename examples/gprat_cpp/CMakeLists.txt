cmake_minimum_required(VERSION 3.16)

project(gprat_cpp)

# This project can be built as part of GPRat itself or as a standalone project
# in which case below find_package() call is used.
if(TARGET GPRat::core)
  message("Using in-tree GPRat::core")
else()
  message("Using out-of-tree GPRat::core")
  find_package(GPRat REQUIRED)
  if(USE_MKL)
    # Try to find Intel oneMKL
    set(MKL_INTERFACE_FULL "intel_lp64")
    set(MKL_THREADING "sequential")
    find_package(MKL CONFIG REQUIRED)
  endif()
endif()

# Enables/disables computation with CUDA on test system
option(GPRAT_WITH_CUDA
       "Enable additional computation with CUDA, cuSolver, cuBLAS" OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_WITH_CUDA=$<BOOL:${GPRAT_WITH_CUDA}>)

# Option for steps duration measurement with APEX
option(GPRAT_APEX_STEPS "Enable measuring duration of steps with APEX" OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_APEX_STEPS=$<BOOL:${GPRAT_APEX_STEPS}>)

# Option for measuring duration of assembly of covariance matrix and right
# looking cholesky in the cholesky function using APEX. Technically, the
# cholesky function also returns a large matrix but this adds significant
# overhead to the whole function call and does not provide much useful
# information on the optimized assembly and right looking cholesky steps which
# are used by other GP operations.
option(
  GPRAT_APEX_CHOLESKY
  "Enable measuring duration of cholesky assembly and computation with APEX"
  OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_APEX_CHOLESKY=$<BOOL:${GPRAT_APEX_CHOLESKY}>)

# Add the executable
add_executable(gprat_cpp src/execute.cpp)

target_compile_features(gprat_cpp PUBLIC cxx_std_17)

# Link the libraries
target_link_libraries(gprat_cpp PUBLIC GPRat::core)
