cmake_minimum_required(VERSION 3.8)

# pickup environment to get vcpkg cmake files
set(vcpkg_root ${OPENPIV_CXX_DIR}/external/vcpkg)
set(VCPKG_TARGET_TRIPLET "${TRIPLET_TO_USE}" CACHE STRING "")
set(CMAKE_TOOL_CHAIN_FILE ${OPENPIV_CXX_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

include(${OPENPIV_CXX_DIR}/cmake/utils.cmake)
setup_vcpkg_before_project()

project(process)

set(CMAKE_CXX_STANDARD 17)

set(process_VERSION_MAJOR 0)
set(process_VERSION_MINOR 1)
set(process_VERSION_PATCH 0)

# set output path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LIB_COPY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${LIB_COPY_DIR})

# get relative path to packages from vcpkg
set(PKG_PATHS ${OPENPIV_CXX_DIR}/external/vcpkg/packages)

# include packages... why can't find_package() not fing configuration file?
find_package(cxxopts PATHS ${PKG_PATHS}/cxxopts_${VCPKG_TARGET_TRIPLET} REQUIRED)
find_package(Async++ PATHS ${PKG_PATHS}/Asyncplusplus_${VCPKG_TARGET_TRIPLET})
find_package(Threads REQUIRED)
find_package(fmt PATHS ${PKG_PATHS}/fmt_${VCPKG_TARGET_TRIPLET} REQUIRED)

# find openpivcore dynamic library
add_library(openpivcore SHARED IMPORTED)
if (WIN32)
    set_target_properties(
       openpivcore
       PROPERTIES
           IMPORTED_LOCATION ${OPENPIV_CXX_DIR}/build/out/Release/openpivcore.dll
           IMPORTED_IMPLIB ${OPENPIV_CXX_DIR}/build/openpiv/Release/openpivcore.lib
    )
elseif (UNIX && APPLE)
    set_target_properties(
           openpivcore
           PROPERTIES
               IMPORTED_LOCATION ${OPENPIV_CXX_DIR}/build/openpiv/libopenpivcore.dylib
        )
elseif (UNIX)
    set_target_properties(
           openpivcore
           PROPERTIES
               IMPORTED_LOCATION ${OPENPIV_CXX_DIR}/build/openpiv/libopenpivcore.so
        )
else()
    message( FATAL_ERROR "Could not locate openpivcore dynamic library.")
endif()

# find pybind11
find_package(pybind11 PATHS ${PYBIND11_DIR} REQUIRED)
pybind11_add_module(_process wrapper.cpp)

# include openpivcore
include_directories(${OPENPIV_CXX_DIR}/openpiv)

add_compile_definitions(_USE_MATH_DEFINES)

target_link_libraries(
  _process
  PRIVATE cxxopts::cxxopts
  PRIVATE fmt::fmt-header-only
  Threads::Threads
  openpivcore)

# include dlls to build
if (WIN32)
    file(GLOB DIST_DLLS "${OPENPIV_CXX_DIR}/build/out/Release/*.dll")
elseif (UNIX && APPLE)
    file(GLOB DIST_DLLS "${OPENPIV_CXX_DIR}/build/openpiv/*.dylib")
elseif (UNIX)
    file(GLOB DIST_DLLS "${OPENPIV_CXX_DIR}/build/openpiv/*.so")
endif()
file(COPY ${DIST_DLLS} DESTINATION ${LIB_COPY_DIR})
