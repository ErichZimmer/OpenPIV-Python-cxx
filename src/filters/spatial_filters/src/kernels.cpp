#include <cmath>
#include <iomanip>

#include "constants.h"
#include "kernels.h"

//-------------CONVOLUTION KERNELS-------------//
imgDtype kernels::apply_conv_kernel(
    const imgDtype* input,
    const std::vector<imgDtype>& kernel,
    int row, int col, int step, 
    int kernel_size
){
    int k_ind{0};
    imgDtype sum{0};
    for (int i{-kernel_size / 2}; i <= (kernel_size / 2); ++i)
    {
        for (int j{-kernel_size / 2}; j <= (kernel_size / 2); ++j)
        {
             // The operation should be done on images with range [0,1]
             sum += kernel[k_ind] * (input[step * (row + i) + (col + j)]);
             ++k_ind;
        }
    }
    return sum;
}

//---------------SORTING KERNELS---------------//