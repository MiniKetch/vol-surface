# ----------------------------------------------------------------------------
# Unit tests via doctest
# ----------------------------------------------------------------------------
# doctest is a single-header testing framework. By default we pull it via
# FetchContent (zero repo bloat). For offline/airgapped/sandboxed builds
# you can set:
#     -DVOL_DOCTEST_INCLUDE_DIR=/path/to/dir-containing-doctest-header
# pointing to a directory whose tree contains  doctest/doctest.h .
# ----------------------------------------------------------------------------

set(VOL_DOCTEST_INCLUDE_DIR "" CACHE PATH
    "Optional path to a vendored doctest header (skip FetchContent if set)")

if(VOL_DOCTEST_INCLUDE_DIR)
    # Vendored / system path: just expose the include directory as an
    # INTERFACE library so the test target gets the right -I.
    add_library(doctest_iface INTERFACE)
    target_include_directories(doctest_iface
        INTERFACE ${VOL_DOCTEST_INCLUDE_DIR})
    add_library(doctest::doctest ALIAS doctest_iface)
    message(STATUS "Tests: using vendored doctest at ${VOL_DOCTEST_INCLUDE_DIR}")
else()
    # doctest's own CMakeLists.txt declares `cmake_minimum_required` at a
    # legacy value that CMake 4.x has dropped support for. We give it a
    # policy-version floor so it configures cleanly under either CMake 3
    # or 4. (This *only* affects the FetchContent-imported subproject —
    # our own CMakeLists already declares 3.20.)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

    include(FetchContent)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.11
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(doctest)
    message(STATUS "Tests: using FetchContent doctest v2.4.11")
endif()

# One test executable bundling all test files. doctest's main is provided
# by defining DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN in exactly one TU.
add_executable(vol_tests
    test_main.cpp
    test_black_scholes.cpp
    test_iv_solver.cpp
    test_delta_solver.cpp
)

target_link_libraries(vol_tests
    PRIVATE
        vol_kernel
        doctest::doctest
)

# Register each TEST_CASE as an individual ctest entry when the discovery
# helper is available (it ships with FetchContent doctest, not always with
# vendored copies).
if(NOT VOL_DOCTEST_INCLUDE_DIR)
    include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
    doctest_discover_tests(vol_tests)
else()
    add_test(NAME vol_tests COMMAND vol_tests)
endif()
