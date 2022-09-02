# include packages... why can't find_package() not find configuration file?
find_package(cxxopts PATHS ${PKG_PATHS}/cxxopts_${VCPKG_TARGET_TRIPLET} REQUIRED)
find_package(Async++ PATHS ${PKG_PATHS}/Asyncplusplus_${VCPKG_TARGET_TRIPLET})
find_package(Threads REQUIRED)
find_package(fmt PATHS ${PKG_PATHS}/fmt_${VCPKG_TARGET_TRIPLET} REQUIRED)

# find openpivcore dynamic library
add_library(openpivcore SHARED IMPORTED)
if (WIN32)
    set_target_properties(openpivcore
       PROPERTIES
           IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/_libs/openpivcore.dll
           IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/_libs/openpivcore.lib
    )
elseif (UNIX && APPLE)
    set_target_properties(openpivcore
        PROPERTIES
           IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/_libs/libopenpivcore.dylib
    )
elseif (UNIX)
    set_target_properties(openpivcore
        PROPERTIES
           IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/_libs/libopenpivcore.so
    )
 else()
     message( FATAL_ERROR "Could not locate openpivcore dynamic library.")
endif()

# include wrapper sources
file (GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# include wrapper sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(_process
    wrapper.cpp
    ${SOURCE_FILES}
)

# include wrapper sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# include openpivcore
include_directories(${OPENPIV_CXX_DIR}/openpiv)

add_compile_definitions(_USE_MATH_DEFINES)

target_link_libraries(_process
    PRIVATE cxxopts::cxxopts
    PRIVATE fmt::fmt-header-only
    Threads::Threads
    openpivcore
)

install(TARGETS _process DESTINATION openpiv_cxx/process)

file(GLOB DIST_DLLS "${PROJECT_SOURCE_DIR}/_libs/*")

install(FILES ${DIST_DLLS} DESTINATION openpiv_cxx/_libs)