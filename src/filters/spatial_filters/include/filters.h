#ifndef FILTER_H
#define FILTER_H

// std
#include <cstdint>
#include <vector>

// utils
#include "constants.h"


using imgDtype = constants::imgDtype;


void intensity_cap_filter(
    imgDtype*,
    imgDtype*,
    std::size_t,
    imgDtype
);


void convolve2D(
    const imgDtype*, 
    imgDtype*, 
    std::uint32_t, 
    std::uint32_t, 
    const imgDtype*,
    const imgDtype*, 
    std::uint32_t
);


#endif