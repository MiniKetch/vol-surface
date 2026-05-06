# ----------------------------------------------------------------------------
# Python extension module via pybind11
# ----------------------------------------------------------------------------
# Built only when VOL_BUILD_PYTHON=ON (set by scikit-build-core during
# `pip install`). Standalone CMake builds skip this so you don't need
# Python/pybind11 installed for development.
# ----------------------------------------------------------------------------

# When invoked via scikit-build-core, both pybind11 and Python are
# already on the CMake module path. As a fallback for direct cmake users
# we try find_package() first.
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# pybind11_add_module wraps Python.h + correct linkage + .so/.pyd naming.
# The output is _vol_kernel.<plat-tag>.so on Linux/macOS or .pyd on Windows.
pybind11_add_module(_vol_kernel MODULE
    python_module.cpp
)

target_link_libraries(_vol_kernel PRIVATE vol_kernel)

# scikit-build-core looks at install() rules to decide what goes in the
# wheel. We install the extension into the vol_surface package directory.
install(TARGETS _vol_kernel
        LIBRARY DESTINATION vol_surface
        RUNTIME DESTINATION vol_surface)
