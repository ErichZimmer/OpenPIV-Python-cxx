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

# setup for building libopenpivcore
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
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -g -O3 -ffast-math -march=native")
endif()

# Set the _using_conda variable when building against a conda environment
get_filename_component(_python_bin_dir ${PYTHON_EXECUTABLE} DIRECTORY)
if(EXISTS "${_python_bin_dir}/../conda-meta")
  message("-- Detected conda environment, setting INSTALL_RPATH_USE_LINK_PATH")
  set(_using_conda On)
else()
  set(_using_conda Off)
endif()

# include pybind11
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)

# build openpivcore
add_subdirectory(${OPENPIV_CXX_DIR}/openpiv)

# include source directory
add_subdirectory(src)

if(_using_conda OR DEFINED ENV{CIBUILDWHEEL})
    set_target_properties(openpivcore PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
endif()

# install built runtime dependencies
if (WIN32)
    set(_EXT "*.dll")
elseif (APPLE)
    set(_EXT "*.dylib")
elseif (UNIX)
    set(_EXT "*.so")
else()
    message(FATAL_ERROR "Could not determine OS for dynamic libraries")
endif()

# Since the submodule process is the only one using shared libraries, place the shared libraries there (for now).
install(DIRECTORY ${CMAKE_BINARY_DIR}/out/
        DESTINATION lib/process
        FILES_MATCHING PATTERN ${_EXT})
        
