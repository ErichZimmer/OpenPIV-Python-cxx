cmake_minimum_required(VERSION 3.8)

# pickup environment to get vcpkg cmake files
set(vcpkg_root ${OPENPIV_CXX_DIR}/external/vcpkg)
set(VCPKG_TARGET_TRIPLET "${TRIPLET_TO_USE}" CACHE STRING "")
set(CMAKE_TOOL_CHAIN_FILE ${OPENPIV_CXX_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

include(${OPENPIV_CXX_DIR}/cmake/utils.cmake)
setup_vcpkg_before_project()

project(openpiv_cxx)

set(CMAKE_CXX_STANDARD 17)
  
set(openpiv_cxx_VERSION_MAJOR 0)
set(openpiv_cxx_VERSION_MINOR 1)
set(openpiv_cxx_VERSION_PATCH 0)

# get relative path to packages from vcpkg
set(PKG_PATHS ${OPENPIV_CXX_DIR}/external/vcpkg/packages)

# find pybind11
find_package(pybind11 PATHS ${PYBIND11_DIR} REQUIRED)

add_subdirectory(src)
