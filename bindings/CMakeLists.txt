# try finding pybind11
set(GPRat_pybind11_VERSION 2.10.3)
find_package(pybind11 ${GPRat_pybind11_VERSION} QUIET)
if(pybind11_FOUND)
  message(STATUS "Found package pybind11.")
else()
  message(STATUS "Couldn't find package pybind11. Building from source ...")
  include(FetchContent)
  set(FETCHCONTENT_QUIET FALSE)
  # fetch pybind11 library for creating Python bindings
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG "v${GPRat_pybind11_VERSION}"
    GIT_SHALLOW TRUE
    QUIET)
  FetchContent_MakeAvailable(pybind11)
  message(STATUS "Installed pybind11 version ${GPRat_pybind11_VERSION}.")
endif()

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")
file(GLOB HEADER_FILES CONFIGURE_DEPENDS "*.hpp")

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

pybind11_add_module(gprat_bindings ${SOURCE_FILES} ${HEADER_FILES})

# must match the Python module name!
set_property(TARGET gprat_bindings PROPERTY OUTPUT_NAME "gprat")

# We'd like to consume it just as `bindings` under the GPRat namespace
set_property(TARGET gprat_bindings PROPERTY EXPORT_NAME bindings)
add_library(GPRat::bindings ALIAS gprat_bindings)

target_link_libraries(gprat_bindings PUBLIC GPRat::core)

if(NOT CMAKE_SKIP_INSTALL_RULES)
  install(
    TARGETS gprat_bindings
    EXPORT GPRatTargets
    RUNTIME COMPONENT Runtime
    LIBRARY COMPONENT Runtime NAMELINK_COMPONENT Development
    ARCHIVE COMPONENT Development
    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    # XXX: Due to pybind11_add_module() CMake doesn't use the defaults from
    # GNUInstallDirs, forcing us to write these out.
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
