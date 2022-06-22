#ifndef CC_UTILS_H
#define CC_UTILS_H

// std
#include <vector>
// #include <cinttypes>
#include <cstddef>

// openpiv
#include "core/rect.h"
#include "core/image.h"
#include "core/image_utils.h"
#include "core/vector.h"

using namespace openpiv;


std::string get_execution_type(int);


core::gf_image placeIntoPadded(
    const core::gf_image&,
    const core::size&,
    const core::rect&,
    double
);


double meanI(
    const core::gf_image&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

std::vector<double> mean_std(
    const core::gf_image&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

        
void applyScalarToImage(
    core::gf_image&,
    double,
    std::size_t
);


void placeIntoCmatrix(
    std::vector<double>&,
    const core::gf_image&,
    const core::size&,
    const core::rect&,
    uint32_t
);

#endif