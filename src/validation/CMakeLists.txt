# include wrapper sources
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

# create library
##add_library(spatFiltPack STATIC ${SOURCE_FILES})

# add wrapper module
pybind11_add_module(_validation 
    wrapper.cpp
)

install(TARGETS _validation DESTINATION openpiv_cxx/validation)