# include external sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext)

add_subdirectory(interpolation)
add_subdirectory(preprocess)
add_subdirectory(process)