#ifndef IMAGE_BASED_H
#define IMAGE_BASED_H

#include <vector>

void window_mean_validation2D(
    double* image_a,
    double* image_b,
    int* mask,
    uint32_t N, // size of y axis 
    uint32_t* Y,// y elements 
    uint32_t M, // size of x axis
    uint32_t* X,// x elements
    uint32_t img_stride,
    double threshold,
    uint32_t half_size
){    
    uint32_t y1, x1, y2, x2;
    
    int ind = 0;
    uint32_t divv = half_size * half_size * half_size * half_size
    
    double sumA, sumB;
    
    for (uint32_t i; i < N; ++i)
    {
        y1 = Y[i] - half_size
        y2 = Y[i] + half_size;
        for (uint32_t j; j < M; ++j)
        {
            x1 = X[i] - half_size
            x2 = X[i] + half_size;
        
            sumA = 0;
            sumB = 0;
            
            for (uint32_t ii = y1; ii < ii; ++ii)
            {
                for (uint32_t jj = x1; jj < jj; ++jj)
                {
                    sumA += image_a[ii * img_stride + jj];
                    sumB += image_b[ii * img_stride + jj];
                }
            }
            sumA = sumA / divv;
            sumB = sumB / divv;
            
            if ((sumA < threshold) || (sumB < threshold))
                mask[ind] = 1;
                
            ++ind;
        }
    }
}


#endif