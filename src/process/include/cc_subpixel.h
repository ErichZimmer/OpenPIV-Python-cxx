#ifndef CC_SUBPIXEL_H
#define CC_SUBPIXEL_H

// std
#include <cinttypes>
#include <vector>

// openpiv
#include "core/image.h"
#include "core/image_utils.h"
#include "core/vector.h"

using namespace openpiv;

core::peaks_t<core::g_f> find_peaks_brute(
    const core::gf_image&,
    uint16_t,
    uint32_t
);


void process_cmatrix_2x3(
    double*,
    double*,
    uint32_t,
    uint32_t,
    core::size,
    int,
    int,
    int
    
);
#endif