# include wrapper sources
file (GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# include wrapper sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(_taylor_expansion2D_cpp
    wrapper.cpp
    ${SOURCE_FILES}
)

install(TARGETS _taylor_expansion2D_cpp DESTINATION lib/interpolate)