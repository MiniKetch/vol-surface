cmake_minimum_required(VERSION 3.20)

# Single source of truth for project version.
project(vol_surface
    VERSION 0.1.0
    DESCRIPTION "Vol Surface + Mispricing Scanner — C++ math kernel + Python orchestration"
    LANGUAGES CXX
)

# ----------------------------------------------------------------------------
# Global C++ standard
# ----------------------------------------------------------------------------
# We pin C++17. It's the modern baseline: structured bindings, std::optional,
# if-init, [[nodiscard]] — all things we'll lean on. C++20 would give us
# concepts and ranges but tightens compiler-version requirements.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # No GNU extensions — keeps us portable.

# ----------------------------------------------------------------------------
# Build options
# ----------------------------------------------------------------------------
option(VOL_BUILD_TESTS "Build C++ unit tests" ON)
option(VOL_BUILD_PYTHON "Build pybind11 Python module" OFF)
option(VOL_WARNINGS_AS_ERRORS "Treat warnings as errors in our code" OFF)

# Default to Release if the user didn't specify. Single-config generators
# (Make, Ninja) need this. Multi-config (Visual Studio, Xcode) ignore it.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Export compile_commands.json so clangd / VS Code IntelliSense work nicely.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# enable_testing() must be called at the top level for ctest to discover
# tests registered in subdirectories.
if(VOL_BUILD_TESTS)
    enable_testing()
endif()

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
add_subdirectory(cpp)

# ----------------------------------------------------------------------------
# Status summary — prints once at configure time so the user knows what
# they're getting.
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "vol_surface ${PROJECT_VERSION} configured")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests:             ${VOL_BUILD_TESTS}")
message(STATUS "  Python bindings:   ${VOL_BUILD_PYTHON}")
message(STATUS "  Warnings as error: ${VOL_WARNINGS_AS_ERRORS}")
message(STATUS "")
