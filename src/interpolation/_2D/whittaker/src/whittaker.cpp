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
    std::uint32_t N,
    std::uint32_t M,
    int radius
){
    int xn=0, yn=0, x=0, y=0;
    int i0=0, i1=0, j0=0, j1=0;
    double dx, dy, bx, by, sx, sy;
    int nMax = static_cast<int>(N) - 1;
    int mMax = static_cast<int>(M) - 1;

    for (std::uint32_t i = 0; i < N; ++i)
    {
        for (std::uint32_t j = 0; j < M; ++j)
        {
            bx = X[i * M + j];
            by = Y[i * M + j];

            xn =  (int) bx;
            yn =  (int) by;

            i0 = xn - radius;
            i1 = xn + radius;
            j0 = yn - radius;
            j1 = yn + radius;

            out[i * M + j] = 0.0;

            // Border policy is nearest value.
            // If we use constants (e.g., zero), the code would likely be much faster.
            for (int k = i0; k <= i1; ++k)
            {
                x = k;
                x = std::max<int>(x, 0);
                x = std::min<int>(x, nMax);

                dx = double(x) - bx;
                sx = sinc(dx);

                for (int h = j0; h <= j1; ++h)
                {
                    y = h;
                    y = std::max<int>(y, 0);
                    y = std::min<int>(y, mMax);

                    dy = double(y) - by;
                    sy = sinc(dy);

                    out[i * M + j] += Z[x * M + y] * sx * sy;
                }
            }
        }
    }
}