cmake_minimum_required(VERSION 3.8)

# pickup environment to get vcpkg cmake files
set(OPENPIV_CXX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/openpiv-c--qt)

set(vcpkg_root ${OPENPIV_CXX_DIR}/external/vcpkg)
set(VCPKG_TARGET_TRIPLET "${TRIPLET_TO_USE}" CACHE STRING "")
set(CMAKE_TOOL_CHAIN_FILE ${OPENPIV_CXX_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

 
include(${OPENPIV_CXX_DIR}/cmake/utils.cmake)
setup_vcpkg_before_project()

project(openpiv_cxx)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
  
set(openpiv_cxx_VERSION_MAJOR 0)
set(openpiv_cxx_VERSION_MINOR 1)
set(openpiv_cxx_VERSION_PATCH 0)

# get relative path to packages from vcpkg
set(PKG_PATHS ${OPENPIV_CXX_DIR}/external/vcpkg/packages)

# include external sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/extras)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)

add_subdirectory(src)
