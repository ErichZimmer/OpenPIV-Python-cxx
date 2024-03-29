#ifndef VECTOR_BASED_H
#define VECTOR_BASED_H

// std
#include <cmath>
#include <cstdint>
#include <vector>


// check 8 points to see if vector in question has less than 50% of points invalid
void difference_test2D(
    const double*,
    const double*,
    int*,
    double,
    double,
    std::uint32_t,
    std::uint32_t
);


double median(
    std::vector<double>
);


void local_median_test(
    const double*,
    const double*,
    int*,
    double,
    double,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::size_t
);


void normalized_local_median_test(
    const double*,
    const double*,
    int*,
    double,
    double,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    double,
    std::size_t
);


double test_median(
    double*,
    size_t
);


#endif