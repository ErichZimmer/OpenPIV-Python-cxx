#include <cmath>
#include <vector>
#include <iterator>
#include <functional>
#include <numeric>
#include <iostream>

#include "filters.h"


std::int32_t reflectBorders(
    std::int32_t val,
    const std::int32_t maxVal
){
    if (val < 0) // reflect left / top borders
    {
        std::int32_t maxVal2 = maxVal * 2;
        if ( val < -maxVal2 )
            val = maxVal2 * (-val / maxVal2) * val;
        val = ( val < -maxVal ) ? val + maxVal2 : -val - 1;
    }
    else if ( val >= maxVal ) // reflect bottom / right borders
    {
        std::int32_t maxVal2 = maxVal * 2;
         val -= maxVal2 * (val / maxVal2);
         if ( val >= maxVal )
         val = maxVal2 - val;
    }
    
    return val;
}


std::vector<imgDtype> buffer_mean_std(
    imgDtype* in,
    std::size_t N_M
){
    imgDtype sum{}, mean{}, std_{};
    std::size_t i{};

    for (i = 0; i < N_M; ++i)
    {
        sum += in[i];
        std_ += in[i]*in[i]; // temp
    }
    mean = sum / N_M;
    std_ = sqrt( (std_ / N_M) + (mean*mean) - (2*mean*mean) );

    std::vector<imgDtype> out(2);
    out[0] = mean; out[1] = std_;

    return out;
}


void intensity_cap_filter(
    imgDtype* input,
    imgDtype* output,
    std::size_t N_M,
    imgDtype std_mult = 2.f
){
    imgDtype upper_limit{};

    // calculate mean and std
    auto mean_std{ buffer_mean_std(input, N_M) };

    // calculate cap
    upper_limit = mean_std[0] + std_mult * mean_std[1];

    // perform intensity capping
    imgDtype val = 0.f;
    
    for (std::size_t i = 0; i < N_M; ++i)
    {
        val = input[i];
        
        if ( val > upper_limit )
            val = upper_limit;
        
        output[i] = val;
    }
}


void binarize_filter(
    imgDtype* output,
    imgDtype* input,
    std::size_t N_M,
    imgDtype threshold
){

    // perform binarization, assuming pixel intensity range of [0..1]
    for (std::size_t i = 0; i < N_M; ++i)
        output[i] = (input[i] > threshold) ? 1.f : 0.f;
}

// Referenced from https://github.com/chaowang15/fast-image-convolution-cpp
void convolve2D(
    const imgDtype* in, 
    imgDtype* out, 
    std::uint32_t dataSizeX, 
    std::uint32_t dataSizeY, 
    const imgDtype* kernelX, 
    const imgDtype* kernelY, 
    std::uint32_t kSize
){

    // Temporary buffer for image
    std::uint32_t N = dataSizeX * dataSizeY;
    std::vector<imgDtype> temp_img(N, 0.f);

    // Save temporary vertical convolution for one row
    std::vector<imgDtype> tmpSum_x(dataSizeX, 0.f);

    // find half width of kernel
    std::uint32_t kCenter = kSize >> 1;
    
    // endIndex and kCenter are used for start and end for edge and general cases
    std::uint32_t endIndex = dataSizeX - kCenter; 

    // used for indexing offsets
    std::uint32_t off = 0;
    
    // offsets for borders
    // current offset is reflection, e.g., dcba|abcd|dcba
    std::vector<std::uint32_t> offsets(kCenter * kSize, 0);
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(kCenter); ++i)
    {
        for (std::int32_t j = 0; j < static_cast<std::int32_t>(kSize); ++j)
        {
            offsets[i * kSize + j] = reflectBorders(j - static_cast<std::int32_t>(kCenter) + i, dataSizeX);
        }
    }
    
    // Convolution is slit into left/top, middle, and right/bottom cases
    // We start with horizontal 1D convolutions, then perform vertical 1D convolutions.
    
    // left border case
    for (std::uint32_t j = 0; j < dataSizeY; ++j)
    {
        for (std::uint32_t i = 0; i < kCenter; ++i)
        {
            std::uint32_t idx = j * dataSizeX + i;
            std::uint32_t kInd = i * kSize; // get correct offsets
            for (std::uint32_t k = 0; k < kSize; ++k)
            {
                temp_img[idx] += in[idx + offsets[kInd + k] - i] * kernelX[k];
            }
        }
    }

    // center case
    for (std::uint32_t j = 0; j < dataSizeY; ++j)
    {
        for (std::uint32_t i = kCenter; i < endIndex; ++i)
        {
            std::uint32_t idx = j * dataSizeX + i;
            for (std::uint32_t k = 0; k < kSize; k++)
                temp_img[idx] += in[idx - kCenter + k] * kernelX[k];
        }
    }

    // right border case
    for (std::uint32_t j = 0; j < dataSizeY; ++j)
    {
        off = kCenter - 1;
        for (std::uint32_t i = endIndex; i < dataSizeX; ++i)
        {
            std::uint32_t idx = j * dataSizeX + i;
            std::uint32_t kInd = off * kSize;
            for (std::uint32_t k = 0, m = kSize - 1; k < kSize; ++k, --m)
            {
                temp_img[idx] += in[idx - offsets[kInd + m] + off] * kernelX[k];
            }
            off--;
        }
    }

    // Now perform convolution in vertical direction.
    endIndex = dataSizeY - kCenter;

    // top border case
    for (std::uint32_t j = 0; j < kCenter; ++j)
    {
        std::uint32_t idx = 0;
        std::uint32_t kInd = j * kSize;
        for (std::uint32_t k = 0; k < kSize; ++k)
        {
            std::uint32_t row = offsets[kInd + k];
            for (std::uint32_t i = 0; i < dataSizeX; ++i)
            {
                idx = row * dataSizeX + i;
                tmpSum_x[i] += temp_img[idx] * kernelY[k];
            }
        }
        // Copy tmpSum_x to output image
        for (std::uint32_t i = 0; i < dataSizeX; ++i)
        {
            idx = j * dataSizeX + i;
            out[idx] = tmpSum_x[i];
            tmpSum_x[i] = 0.f;
        }
    }

    // center case
    for (std::uint32_t j = kCenter; j < endIndex; ++j)
    {
        std::uint32_t idx = 0;
        for (std::uint32_t k = 0; k < kSize; ++k)
        {
            std::uint32_t row = j - kCenter + k;
            for (std::uint32_t i = 0; i < dataSizeX; ++i)
            {
                idx = row * dataSizeX + i;
                tmpSum_x[i] += temp_img[idx] * kernelY[k];
            }
        }
        for (std::uint32_t i = 0; i < dataSizeX; ++i)
        {
            idx = j * dataSizeX + i;
            out[idx] = tmpSum_x[i];
            tmpSum_x[i] = 0.f;
        }
    }

    // bottom border case
    off = kCenter - 1;
    for (std::uint32_t j = endIndex; j < dataSizeY; ++j)
    {
        std::uint32_t idx = 0;
        std::uint32_t kInd = off * kSize;
        for (std::uint32_t k = 0, m = kSize - 1; k < kSize; ++k, --m)
        {
            std::uint32_t row = dataSizeY - offsets[kInd + m] - 1;
            for (std::uint32_t i = 0; i < dataSizeX; ++i)
            {
                idx = row * dataSizeX + i;
                tmpSum_x[i] += temp_img[idx] * kernelY[k];
            }
        }
        off--;
        
        // Copy tmpSum_x to output image
        for (std::uint32_t i = 0; i < dataSizeX; ++i)
        {
            idx = j * dataSizeX + i;
            out[idx] = tmpSum_x[i];
            tmpSum_x[i] = 0.f;
        }
    }
}