#ifndef KERNELS_H
#define KERNELS_H

#include <vector>
#include <functional>

#include "constants.h"

using imgDtype = constants::imgDtype;

namespace kernels
{
    imgDtype apply_conv_kernel(
        const imgDtype*,
        const std::vector<imgDtype>&,
        int, int, int,
        int
    );
}

#endif