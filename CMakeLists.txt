cmake_minimum_required(VERSION 3.8)

# pickup environment to get vcpkg cmake files
set(OPENPIV_CXX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/openpiv-c--qt)
set(vcpkg_root ${OPENPIV_CXX_DIR}/external/vcpkg)
include(${OPENPIV_CXX_DIR}/cmake/utils.cmake)
setup_vcpkg_before_project()

project(openpiv_cxx)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
  
set(openpiv_cxx_VERSION_MAJOR 0)
set(openpiv_cxx_VERSION_MINOR 1)
set(openpiv_cxx_VERSION_PATCH 0)

# include external sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/extras)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)

# setup for building libopenpivcore0
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(WIN32)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)

if(MSVC)
  set(CMAKE_CXX_FLAGS "/EHsc")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -g -O3 -ffast-math")
endif()

# build libopenpiv
add_subdirectory(${OPENPIV_CXX_DIR}/openpiv)

# include source directory
add_subdirectory(src)

# install runtime dependencies
if (WIN32)
    set(_EXT "*.dll")
elseif (APPLE)
    set(_EXT "*.dylib")
elseif (UNIX)
    set(_EXT "*.so")
else()
    message(FATAL_ERROR "Could not determine OS for dynamic libraries")
endif()

##file(GLOB DIST_DLLS ${CMAKE_BINARY_DIR}/out/*.${_EXT})

message(STATUS "Files ${CMAKE_BINARY_DIR}/out/*.${_EXT} copied to _libs")
install(DIRECTORY ${CMAKE_BINARY_DIR}/out/
        DESTINATION openpiv_cxx/_libs
        FILES_MATCHING PATTERN ${_EXT})