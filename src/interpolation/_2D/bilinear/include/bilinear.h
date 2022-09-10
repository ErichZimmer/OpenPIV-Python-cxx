#ifndef INTERP_BILINEAR_GRIDDED
#define INTERP_BILINEAR_GRIDDED

// std
#include <cstdint>


uint32_t find_index(
    const uint32_t*,
    double,
    uint32_t, // lower bound
    uint32_t  // upper bound
);


void bilinear2D(
    const int*,
    const int*,
    const double*,
    const double*,
    const double*,
    double*,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t
);


#endif