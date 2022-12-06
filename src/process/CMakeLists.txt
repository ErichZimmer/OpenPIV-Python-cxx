# find packages
find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)

# include wrapper sources
file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# include wrapper headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# include openpivcore headers
include_directories(${OPENPIV_CXX_DIR}/openpiv)

# include pocketfft headers
include_directories(${OPENPIV_CXX_DIR}/external/pocketfft)

# include threading headers
include_directories(${PROJECT_SOURCE_DIR}/extern/extras)

# create module
pybind11_add_module(_process_cpp
    wrapper.cpp
    ${SOURCE_FILES}
)

# used in libopenpivcore
add_compile_definitions(_USE_MATH_DEFINES)

# link libraries
target_link_libraries(_process_cpp
    PRIVATE fmt::fmt-header-only
    Threads::Threads
    openpivcore
)

# used in pocketfft
target_compile_definitions(_process_cpp PRIVATE
    POCKETFFT_NO_MULTITHREADING
    POCKETFFT_NO_VECTORS
)

# install to namespace package
install(TARGETS _process_cpp DESTINATION lib/process)

# inspired from Freud
if(APPLE)
    set_target_properties(_process_cpp PROPERTIES INSTALL_RPATH "@loader_path")
else()
    set_target_properties(_process_cpp PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if(_using_conda OR DEFINED ENV{CIBUILDWHEEL})
    set_target_properties(_process_cpp
        PROPERTIES INSTALL_RPATH_USE_LINK_PATH True
    )
endif()