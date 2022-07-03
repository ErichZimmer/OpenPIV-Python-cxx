#ifndef INTERP_WHITTAKER_GRIDDED
#define INTERP_WHITTAKER_GRIDDED

//std
#include <cstdint>

double sinc(
    double
);


void whittaker2D(
    const double*,
    const double*,
    const double*,
    double*,
    uint32_t,
    uint32_t,
    int
);

#endif