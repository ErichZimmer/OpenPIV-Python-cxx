#ifndef KERNELS_H
#define KERNELS_H

#include <vector>
#include <functional>

#include "constants.h"

using imgDtype = constants::imgDtype;

namespace kernels
{
    std::vector<imgDtype> gaussian(int, imgDtype);
    std::vector<imgDtype> box(int, imgDtype);

    std::function<std::vector<imgDtype>(int, imgDtype)> get_kernel_type(int);

    imgDtype apply_conv_kernel(
        const imgDtype*,
        const std::vector<imgDtype>&,
        int, int, int,
        int
    );
}

#endif