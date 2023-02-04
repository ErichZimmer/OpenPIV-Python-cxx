#ifndef INTERP_WHITTAKER_GRIDDED
#define INTERP_WHITTAKER_GRIDDED

// std
#include <cstdint>


double sinc(
    double
);


void whittaker2D(
    const double*,
    const double*,
    const double*,
    double*,
    std::uint32_t,
    std::uint32_t,
    int
);


#endif