# include wrapper sources
file (GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# include wrapper sources
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# add wrapper module
pybind11_add_module(_spatial_filters 
    wrapper.cpp
    ${SOURCE_FILES}
)

install(TARGETS _spatial_filters DESTINATION openpiv_cxx/preprocess)