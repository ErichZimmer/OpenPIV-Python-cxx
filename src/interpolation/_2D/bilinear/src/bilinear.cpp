// std
#include <vector>
#include <iostream>

// interp
#include "bilinear.h"

    
std::uint32_t find_index(
    const int* arr,
    double x,
    std::uint32_t ub  // upper bound
){
    int mid = 0;
    std::uint32_t lb = 0;

    while (lb < ub)
    {
        mid = (lb + ub) / 2;
        
        if ( arr[mid] < x )
            lb = mid + 1;
        else
            ub = mid;
    }

    return lb;
}


void bilinear2D(
    const int* X,
    const int* Y,
    const double* Z,
    const double* xi,
    const double* yi,
    double* out,
    std::uint32_t N,
    std::uint32_t M,
    std::uint32_t img_step,
    std::uint32_t xUpperBound,
    std::uint32_t yUpperBound
){
    int y1, y2, x1, x2;
    double y, x, z11, z12, z21, z22;
    std::uint32_t y_ind, x_ind, ii, jj;
    
    for (std::uint32_t i = 0; i < N; ++i)
    {
        x_ind = find_index(X, xi[i], xUpperBound);

        ii = i;
        if (x_ind == 0)
        {
            x_ind = 1; // ignore edges
            ii = ii + 1;
        }

        x1 = X[x_ind - 1];
        x2 = X[x_ind];
        x  = xi[ii];

        for (std::uint32_t j = 0; j < M; ++j)
        {
            y_ind = find_index(Y, yi[j], yUpperBound);

            jj = j;
            if (y_ind == 0)
            {
                y_ind = 1; // ignore edges
                jj = jj + 1;
            }

//            std::cout << i << ' ' << j << ' ' << xi[i] << ' ' << yi[j] << '\n';

//            std::cout << i << ' ' << j << ' ' << x_ind << ' ' << y_ind << '\n';

            y1 = Y[y_ind - 1];
            y2 = Y[y_ind];
            y  = yi[jj];

//            std::cout << x1 << ' ' << x2 << ' ' << y1 << ' ' << y2 << '\n';

            z11 = Z[(y_ind - 1) * img_step + x_ind - 1];
            z12 = Z[(y_ind - 1) * img_step + x_ind];
            z21 = Z[y_ind * img_step + x_ind - 1];
            z22 = Z[y_ind * img_step + x_ind];

//            std::cout << z11 << ' ' << z12 << ' ' << z21 << ' ' << z22 << '\n';

            out[j * N + i] = (
                (z11 * (x2 - x) * (y2 - y) +
                 z12 * (x - x1) * (y2 - y) +
                 z21 * (x2 - x) * (y - y1) +
                 z22 * (x - x1) * (y - y1) ) / ((x2 - x1) * (y2 - y1))
            );
        }
    }
}