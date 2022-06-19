#ifndef CC_UTILS_H
#define CC_UTILS_H

// std
#include <vector>
#include <iostream>
#include <string>
#include <cinttypes>

// openpiv
#include "core/rect.h"
#include "core/image.h"
#include "core/image_utils.h"
#include "core/vector.h"

using namespace openpiv;


std::string get_execution_type(int);


core::gf_image placeIntoPadded(
    core::gf_image&,
    core::size&,
    const core::rect&,
    double
);


double meanI(
    core::gf_image&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

std::vector<double> mean_std(
    core::gf_image&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

        
void applyScalarToImage(
    core::gf_image&,
    double,
    ssize_t
);


void placeIntoCmatrix(
    std::vector<double>&,
    core::gf_image,
    core::size,
    core::rect,
    uint32_t
);

#endif