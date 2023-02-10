#ifndef INTERP_BILINEAR_GRIDDED
#define INTERP_BILINEAR_GRIDDED

// std
#include <cstdint>


std::uint32_t find_index(
    const int*,
    double,
    std::uint32_t  // upper bound
);


void bilinear2D(
    const int*,
    const int*,
    const double*,
    const double*,
    const double*,
    double*,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t
);


#endif