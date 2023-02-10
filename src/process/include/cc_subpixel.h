#ifndef CC_SUBPIXEL_H
#define CC_SUBPIXEL_H

// std
#include <cinttypes>
#include <vector>

// openpiv
#include "core/image.h"
#include "core/image_utils.h"
#include "core/vector.h"

// utils
#include "constants.h"


using imgDtype = constants::imgDtype;
using namespace openpiv;


core::image<core::g<imgDtype>> find_peaks_brute(
    const core::gf_image&,
    std::uint16_t,
    std::uint32_t
);


void process_cmatrix_2x3(
    const imgDtype*,
    double*,
    std::uint32_t,
    std::uint32_t,
    std::vector<std::uint32_t>,
    int,
    int,
    int
    
);
#endif