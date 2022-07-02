# include wrapper sources
file (GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# include wrapper sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(_whittaker2D
    wrapper.cpp
    ${SOURCE_FILES}
)

install(TARGETS _whittaker2D DESTINATION openpiv_cxx/interpolate)