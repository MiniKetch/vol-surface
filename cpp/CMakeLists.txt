# ----------------------------------------------------------------------------
# vol_kernel — the C++ math library
# ----------------------------------------------------------------------------
# We split this into a library (vol_kernel) so it can be linked by:
#   - the test executable (so tests don't recompile the kernel sources)
#   - the pybind11 Python module
#   - any future C++ consumer
#
# STATIC for now — small, zero deployment footprint, no DLL/dylib runtime
# linker dance. Switch to OBJECT or SHARED later if the kernel grows or we
# want to share a single .dylib across Python modules.
# ----------------------------------------------------------------------------

add_library(vol_kernel STATIC
    src/black_scholes.cpp
    src/iv_solver.cpp
    src/delta_solver.cpp
)

# Public headers live under cpp/include/ so consumers `#include <vol/...>`.
target_include_directories(vol_kernel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Position-Independent Code: required for linking a static lib into a
# Python shared module on Linux. Harmless on macOS and Windows.
set_target_properties(vol_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ----------------------------------------------------------------------------
# Warnings — strict in our code, lenient in third-party (FetchContent etc.)
# ----------------------------------------------------------------------------
if(MSVC)
    target_compile_options(vol_kernel PRIVATE /W4 /permissive-)
    if(VOL_WARNINGS_AS_ERRORS)
        target_compile_options(vol_kernel PRIVATE /WX)
    endif()
else()
    target_compile_options(vol_kernel PRIVATE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wconversion -Wsign-conversion
        -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused -Woverloaded-virtual
        -Wnull-dereference -Wdouble-promotion
        -Wformat=2
    )
    if(VOL_WARNINGS_AS_ERRORS)
        target_compile_options(vol_kernel PRIVATE -Werror)
    endif()
endif()

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(VOL_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Python bindings (built only when invoked via scikit-build-core)
# ----------------------------------------------------------------------------
if(VOL_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()
