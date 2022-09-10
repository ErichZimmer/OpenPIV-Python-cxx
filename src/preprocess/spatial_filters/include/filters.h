#ifndef FILTER_H
#define FILTER_H

// std
#include <vector>

// utils
#include "constants.h"


using imgDtype = constants::imgDtype;

void intensity_cap_filter(
    imgDtype*,
    int,
    imgDtype
);

void binarize_filter(
    imgDtype*,
    imgDtype*,
    int,
    imgDtype
);

void apply_kernel_lowpass(
    imgDtype*,
    imgDtype*,
    std::vector<imgDtype>&,
    int, int,
    int
);

void apply_kernel_highpass(
    imgDtype*,
    imgDtype*,
    std::vector<imgDtype>&,
    int, int,
    int,
    bool
);

void local_variance_norm(
    imgDtype*,
    imgDtype*,
    imgDtype*,
    int, int,
    int,
    imgDtype,
    imgDtype,
    bool
);


#endif