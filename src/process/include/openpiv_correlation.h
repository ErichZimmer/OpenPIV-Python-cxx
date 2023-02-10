#ifndef CC_CROSSCORRELATE_H
#define CC_CROSSCORRELATE_H

// std
#include <cinttypes>
#include <vector>

// openpiv
#include "core/image.h"

// utils
#include "constants.h"


using namespace openpiv;
using imgDtype = constants::imgDtype;


std::vector<imgDtype> process_window(
    const core::image<core::g<imgDtype>>&,
    const core::image<core::g<imgDtype>>&
);


std::vector<imgDtype> images_to_correlation_standard(
    const core::image<core::g<imgDtype>>&,
    const core::image<core::g<imgDtype>>&,
    std::uint32_t,
    std::uint32_t,
    int,
    int
);


void correlation_based_correction(
    imgDtype*,
    imgDtype*,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    int
);


#endif