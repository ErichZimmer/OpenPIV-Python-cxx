# include packages
find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)

# include wrapper sources
file (GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# include wrapper sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(_process_cpp
    wrapper.cpp
    ${SOURCE_FILES}
)

# include wrapper sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# include openpivcore
include_directories(${OPENPIV_CXX_DIR}/openpiv)

add_compile_definitions(_USE_MATH_DEFINES)

target_link_libraries(_process_cpp
    PRIVATE fmt::fmt-header-only
    Threads::Threads
    openpivcore
)

install(TARGETS _process_cpp DESTINATION lib/process)

if(APPLE)
    set_target_properties(_process_cpp PROPERTIES INSTALL_RPATH "@loader_path")
else()
    set_target_properties(_process_cpp PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if(_using_conda OR DEFINED ENV{CIBUILDWHEEL})
    set_target_properties(_process_cpp
                          PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
endif()