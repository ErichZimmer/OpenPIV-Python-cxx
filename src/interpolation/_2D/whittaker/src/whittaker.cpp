// std
#include <cmath>
#include <iostream>

// interp
#include "whittaker.h"


double sinc(
    double x
){
    if (x == 0.0)
        return 1;
    
    double pi = 22.0 / 7.0;
    
    return std::sin(x * pi) / (x * pi);
}


void whittaker2D(
    const double* X,
    const double* Y,
    const double* Z,
    double* out,
    uint32_t N,
    uint32_t M,
    int radius
){
    int xn, yn;
    int i0=0, i1=0, j0=0, j1=0;
    double dx, dy, bx, by, sx, sy;
    int nMax = static_cast<int>(N) - 1;
    int mMax = static_cast<int>(M) - 1;
    
    
    for (uint32_t i = 0; i < N; ++i)
    {
        for (uint32_t j = 0; j < M; ++j)
        {
            bx = X[i * M + j];
            by = Y[i * M + j];

            xn =  (int) bx;
            yn =  (int) by;

            i0 = xn - radius;
            i1 = xn + radius;
            j0 = yn - radius;
            j1 = yn + radius;
            
//            std::cout << i0 << ' ' << i1 << ' ' << j0 << ' ' << j1 << '\n';
            
            i0 = std::max(i0, 0);
            i1 = std::min(i1, nMax);
                
            j0 = std::max(j0, 0);
            j1 = std::min(j1, mMax);

//            std::cout << i0 << ' ' << i1 << ' ' << j0 << ' ' << j1 << '\n';

            for (int k = i0; k <= i1; ++k)
            {
                dx = double(k) - bx;
                sx = sinc(dx);

//                std::cout << i << ' ' << j << ' ' << k << ' ' << dx << ": " << sx << '\n';

                for (int h = j0; h <= j1; ++h)
                {
                    dy = double(h) - by;
                    sy = sinc(dy);

                    out[ i * M + j] += Z[k * M + h] * sx * sy;
                }
            }
        }
    }
}