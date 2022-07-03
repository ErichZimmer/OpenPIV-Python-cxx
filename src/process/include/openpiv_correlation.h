#ifndef CC_NCC_H
#define CC_NCC_H

// std
#include <cinttypes>
#include <vector>

// openpiv
#include "core/image.h"

using namespace openpiv;


std::vector<double> process_images_autocorrelate(
    core::gf_image&,
    uint32_t,
    uint32_t,
    int,
    int
);


std::vector<double> process_images_scc(
    core::gf_image&,
    core::gf_image&,
    uint32_t,
    uint32_t,
    int,
    int
);


std::vector<double> process_images_ncc(
    core::gf_image&,
    core::gf_image&,
    uint32_t,
    uint32_t,
    int,
    int
);

#endif