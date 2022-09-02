#ifndef INTERP_TAYLOR_EXPANSION_GRIDDED
#define INTERP_TAYLOR_EXPANSION_GRIDDED

//std
#include <cstdint>

void taylor_expansion_k1_2D(
    const double*,
    const double*,
    const double*,
    double*,
    std::uint32_t,
    std::uint32_t
);


void taylor_expansion_k3_2D(
    const double*,
    const double*,
    const double*,
    double*,
    std::uint32_t,
    std::uint32_t
);


void taylor_expansion_k5_2D(
    const double*,
    const double*,
    const double*,
    double*,
    std::uint32_t,
    std::uint32_t
);


void taylor_expansion_k7_2D(
    const double*,
    const double*,
    const double*,
    double*,
    std::uint32_t,
    std::uint32_t
);

#endif