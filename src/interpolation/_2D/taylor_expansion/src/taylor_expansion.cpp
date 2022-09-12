// std
#include <cmath>
#include <iostream>
#include <vector>

// interp
#include "taylor_expansion.h"


void taylor_expansion_k1_2D(
    const double* X,
    const double* Y,
    const double* Z,
    double* out,
    std::uint32_t N,
    std::uint32_t M
){
    int xn, yn, ixi, iyi = 0;
    double bx, by, ratx, raty = 0.0;
    int nMax = static_cast<int>(N) - 1;
    int mMax = static_cast<int>(M) - 1;
    std::vector<double> asx(2, 0), asy(2, 0);
    
    for (std::uint32_t i = 0; i < N; ++i)
    {
        for (std::uint32_t j = 0; j < M; ++j)
        {
            bx = X[i * M + j];
            by = Y[i * M + j];
            
//            std::cout << "bx: " << bx << " by: " << by << '\n';

            xn = static_cast<int>(bx);
            yn = static_cast<int>(by);
            
//            std::cout << "xn: " << xn << " yn: " << yn << '\n';

            ratx = bx - (static_cast<double>(xn) + 0.5);
            raty = by - (static_cast<double>(yn) + 0.5);

//            std::cout << "ratx: " << ratx << " raty: " << raty << '\n';

            asx[0] = 0.5 - ratx;
            asx[1] = 0.5 + ratx;
            asy[0] = 0.5 - raty;
            asy[1] = 0.5 + raty;

            out[i * M + j] = 0.0;

            for (int k = 0; k < 2; ++k)
            {
                ixi = std::max(xn + k, 0);
                ixi = std::min(ixi, nMax);

                for (int m = 0; m < 2; ++m)
                {
                    iyi = std::max(yn + m, 0);
                    iyi = std::min(iyi, mMax);

                    out[i * M + j] += Z[ixi * M + iyi] * asx[k] * asy[m];
                }
            }
        }
    }
}


void taylor_expansion_k3_2D(
    const double* X,
    const double* Y,
    const double* Z,
    double* out,
    std::uint32_t N,
    std::uint32_t M
){
    int xn, yn, ixi, iyi = 0;
    double bx, by, ratx, raty = 0.0;
    int nMax = static_cast<int>(N) - 1;
    int mMax = static_cast<int>(M) - 1;
    std::vector<double> asx(4, 0), asy(4, 0);

    for (std::uint32_t i = 0; i < N; ++i)
    {
        for (std::uint32_t j = 0; j < M; ++j)
        {
            bx = X[i * M + j];
            by = Y[i * M + j];

            xn = static_cast<int>(bx);
            yn = static_cast<int>(by);

            ratx = bx - (static_cast<double>(xn) + 0.5);
            raty = by - (static_cast<double>(yn) + 0.5);

            asx[0] = -1.0/16.0 + ratx*( 1.0/24.0 + ratx*( 1.0/4.0 - ratx/6.0));
            asx[1] =  9.0/16.0 + ratx*( -9.0/8.0 + ratx*(-1.0/4.0 + ratx/2.0));
            asx[2] =  9.0/16.0 + ratx*(  9.0/8.0 + ratx*(-1.0/4.0 - ratx/2.0));
            asx[3] = -1.0/16.0 + ratx*(-1.0/24.0 + ratx*( 1.0/4.0 + ratx/6.0));

            asy[0] = -1.0/16.0 + raty*( 1.0/24.0 + raty*( 1.0/4.0 - raty/6.0));
            asy[1] =  9.0/16.0 + raty*( -9.0/8.0 + raty*(-1.0/4.0 + raty/2.0));
            asy[2] =  9.0/16.0 + raty*(  9.0/8.0 + raty*(-1.0/4.0 - raty/2.0));
            asy[3] = -1.0/16.0 + raty*(-1.0/24.0 + raty*( 1.0/4.0 + raty/6.0));

            xn -= 1;
            yn -= 1;

            out[i * M + j] = 0.0;

            for (int k = 0; k < 4; ++k)
            {
                ixi = std::max(xn + k, 0);
                ixi = std::min(ixi, nMax);

                for (int m = 0; m < 4; ++m)
                {
                    iyi = std::max(yn + m, 0);
                    iyi = std::min(iyi, mMax);

                    out[i * M + j] += Z[ixi * M + iyi] * asx[k] * asy[m];
                }
            }
        }
    }
}


void taylor_expansion_k5_2D(
    const double* X,
    const double* Y,
    const double* Z,
    double* out,
    std::uint32_t N,
    std::uint32_t M
){
    int xn, yn, ixi, iyi = 0;
    double bx, by, ratx, raty = 0.0;
    int nMax = static_cast<int>(N) - 1;
    int mMax = static_cast<int>(M) - 1;
    std::vector<double> asx(6, 0), asy(6, 0);
    
    for (std::uint32_t i = 0; i < N; ++i)
    {
        for (std::uint32_t j = 0; j < M; ++j)
        {
            bx = X[i * M + j];
            by = Y[i * M + j];
            
            xn = static_cast<int>(bx);
            yn = static_cast<int>(by);
            
            ratx = bx - (static_cast<double>(xn) + 0.5);
            raty = by - (static_cast<double>(yn) + 0.5);

            asx[0] =   3.0/256.0 + ratx*(   -9.0/1920.0 + ratx*( -5.0/48.0/2.0 + ratx*(  1.0/8.0/6.0 + ratx*( 1.0/2.0/24.0 -  1.0/8.0/120.0*ratx))));
            asx[1] = -25.0/256.0 + ratx*(  125.0/1920.0 + ratx*( 39.0/48.0/2.0 + ratx*(-13.0/8.0/6.0 + ratx*(-3.0/2.0/24.0 +  5.0/8.0/120.0*ratx))));
            asx[2] = 150.0/256.0 + ratx*(-2250.0/1920.0 + ratx*(-34.0/48.0/2.0 + ratx*( 34.0/8.0/6.0 + ratx*( 2.0/2.0/24.0 - 10.0/8.0/120.0*ratx))));
            asx[3] = 150.0/256.0 + ratx*( 2250.0/1920.0 + ratx*(-34.0/48.0/2.0 + ratx*(-34.0/8.0/6.0 + ratx*( 2.0/2.0/24.0 + 10.0/8.0/120.0*ratx))));
            asx[4] = -25.0/256.0 + ratx*( -125.0/1920.0 + ratx*( 39.0/48.0/2.0 + ratx*( 13.0/8.0/6.0 + ratx*(-3.0/2.0/24.0 -  5.0/8.0/120.0*ratx))));
            asx[5] =   3.0/256.0 + ratx*(    9.0/1920.0 + ratx*( -5.0/48.0/2.0 + ratx*( -1.0/8.0/6.0 + ratx*( 1.0/2.0/24.0 +  1.0/8.0/120.0*ratx))));

            asy[0] =   3.0/256.0 + raty*(   -9.0/1920.0 + raty*( -5.0/48.0/2.0 + raty*(  1.0/8.0/6.0 + raty*( 1.0/2.0/24.0 -  1.0/8.0/120.0*raty))));
            asy[1] = -25.0/256.0 + raty*(  125.0/1920.0 + raty*( 39.0/48.0/2.0 + raty*(-13.0/8.0/6.0 + raty*(-3.0/2.0/24.0 +  5.0/8.0/120.0*raty))));
            asy[2] = 150.0/256.0 + raty*(-2250.0/1920.0 + raty*(-34.0/48.0/2.0 + raty*( 34.0/8.0/6.0 + raty*( 2.0/2.0/24.0 - 10.0/8.0/120.0*raty))));
            asy[3] = 150.0/256.0 + raty*( 2250.0/1920.0 + raty*(-34.0/48.0/2.0 + raty*(-34.0/8.0/6.0 + raty*( 2.0/2.0/24.0 + 10.0/8.0/120.0*raty))));
            asy[4] = -25.0/256.0 + raty*( -125.0/1920.0 + raty*( 39.0/48.0/2.0 + raty*( 13.0/8.0/6.0 + raty*(-3.0/2.0/24.0 -  5.0/8.0/120.0*raty))));
            asy[5] =   3.0/256.0 + raty*(    9.0/1920.0 + raty*( -5.0/48.0/2.0 + raty*( -1.0/8.0/6.0 + raty*( 1.0/2.0/24.0 +  1.0/8.0/120.0*raty))));

            xn -= 2;
            yn -= 2;

            out[i * M + j] = 0.0;

            for (int k = 0; k < 6; ++k)
            {
                ixi = std::max(xn + k, 0);
                ixi = std::min(ixi, nMax);

                for (int m = 0; m < 6; ++m)
                {
                    iyi = std::max(yn + m, 0);
                    iyi = std::min(iyi, mMax);

                    out[i * M + j] += Z[ixi * M + iyi] * asx[k] * asy[m];
                }
            }
        }
    }
}


void taylor_expansion_k7_2D(
    const double* X,
    const double* Y,
    const double* Z,
    double* out,
    std::uint32_t N,
    std::uint32_t M
){
    int xn, yn, ixi, iyi = 0;
    double bx, by, ratx, raty = 0.0;
    int nMax = static_cast<int>(N) - 1;
    int mMax = static_cast<int>(M) - 1;
    std::vector<double> asx(8, 0), asy(8, 0);
    
    for (std::uint32_t i = 0; i < N; ++i)
    {
        for (std::uint32_t j = 0; j < M; ++j)
        {
            bx = X[i * M + j];
            by = Y[i * M + j];
            
            xn = static_cast<int>(bx);
            yn = static_cast<int>(by);
            
            ratx = bx - (static_cast<double>(xn) + 0.5);
            raty = by - (static_cast<double>(yn) + 0.5);

            asx[0] =   -5.0/2048.0 + ratx*(     75.0/107520.0 + ratx*(  259.0/11520.0/2.0 + ratx*(  -37.0/1920.0/6.0 + ratx*(  -7.0/48.0/24.0 + ratx*(   5.0/24.0/120.0 + ratx*( 1.0/2.0/720.0 -  1.0/5040.0*ratx))))));
            asx[1] =   49.0/2048.0 + ratx*(  -1029.0/107520.0 + ratx*(-2495.0/11520.0/2.0 + ratx*(  499.0/1920.0/6.0 + ratx*(  59.0/48.0/24.0 + ratx*( -59.0/24.0/120.0 + ratx*(-5.0/2.0/720.0 +  7.0/5040.0*ratx))))));
            asx[2] = -245.0/2048.0 + ratx*(   8575.0/107520.0 + ratx*(11691.0/11520.0/2.0 + ratx*(-3897.0/1920.0/6.0 + ratx*(-135.0/48.0/24.0 + ratx*( 225.0/24.0/120.0 + ratx*( 9.0/2.0/720.0 - 21.0/5040.0*ratx))))));
            asx[3] = 1225.0/2048.0 + ratx*(-128625.0/107520.0 + ratx*(-9455.0/11520.0/2.0 + ratx*( 9455.0/1920.0/6.0 + ratx*(  83.0/48.0/24.0 + ratx*(-415.0/24.0/120.0 + ratx*(-5.0/2.0/720.0 + 35.0/5040.0*ratx))))));
            asx[4] = 1225.0/2048.0 + ratx*( 128625.0/107520.0 + ratx*(-9455.0/11520.0/2.0 + ratx*(-9455.0/1920.0/6.0 + ratx*(  83.0/48.0/24.0 + ratx*( 415.0/24.0/120.0 + ratx*(-5.0/2.0/720.0 - 35.0/5040.0*ratx))))));
            asx[5] = -245.0/2048.0 + ratx*(  -8575.0/107520.0 + ratx*(11691.0/11520.0/2.0 + ratx*( 3897.0/1920.0/6.0 + ratx*(-135.0/48.0/24.0 + ratx*(-225.0/24.0/120.0 + ratx*( 9.0/2.0/720.0 + 21.0/5040.0*ratx))))));
            asx[6] =   49.0/2048.0 + ratx*(   1029.0/107520.0 + ratx*(-2495.0/11520.0/2.0 + ratx*( -499.0/1920.0/6.0 + ratx*(  59.0/48.0/24.0 + ratx*(  59.0/24.0/120.0 + ratx*(-5.0/2.0/720.0 -  7.0/5040.0*ratx))))));
            asx[7] =   -5.0/2048.0 + ratx*(    -75.0/107520.0 + ratx*(  259.0/11520.0/2.0 + ratx*(   37.0/1920.0/6.0 + ratx*(  -7.0/48.0/24.0 + ratx*(  -5.0/24.0/120.0 + ratx*( 1.0/2.0/720.0 +  1.0/5040.0*ratx))))));

            asy[0] =   -5.0/2048.0 + raty*(     75.0/107520.0 + raty*(  259.0/11520.0/2.0 + raty*(  -37.0/1920.0/6.0 + raty*(  -7.0/48.0/24.0 + raty*(   5.0/24.0/120.0 + raty*( 1.0/2.0/720.0 -  1.0/5040.0*raty))))));
            asy[1] =   49.0/2048.0 + raty*(  -1029.0/107520.0 + raty*(-2495.0/11520.0/2.0 + raty*(  499.0/1920.0/6.0 + raty*(  59.0/48.0/24.0 + raty*( -59.0/24.0/120.0 + raty*(-5.0/2.0/720.0 +  7.0/5040.0*raty))))));
            asy[2] = -245.0/2048.0 + raty*(   8575.0/107520.0 + raty*(11691.0/11520.0/2.0 + raty*(-3897.0/1920.0/6.0 + raty*(-135.0/48.0/24.0 + raty*( 225.0/24.0/120.0 + raty*( 9.0/2.0/720.0 - 21.0/5040.0*raty))))));
            asy[3] = 1225.0/2048.0 + raty*(-128625.0/107520.0 + raty*(-9455.0/11520.0/2.0 + raty*( 9455.0/1920.0/6.0 + raty*(  83.0/48.0/24.0 + raty*(-415.0/24.0/120.0 + raty*(-5.0/2.0/720.0 + 35.0/5040.0*raty))))));
            asy[4] = 1225.0/2048.0 + raty*( 128625.0/107520.0 + raty*(-9455.0/11520.0/2.0 + raty*(-9455.0/1920.0/6.0 + raty*(  83.0/48.0/24.0 + raty*( 415.0/24.0/120.0 + raty*(-5.0/2.0/720.0 - 35.0/5040.0*raty))))));
            asy[5] = -245.0/2048.0 + raty*(  -8575.0/107520.0 + raty*(11691.0/11520.0/2.0 + raty*( 3897.0/1920.0/6.0 + raty*(-135.0/48.0/24.0 + raty*(-225.0/24.0/120.0 + raty*( 9.0/2.0/720.0 + 21.0/5040.0*raty))))));
            asy[6] =   49.0/2048.0 + raty*(   1029.0/107520.0 + raty*(-2495.0/11520.0/2.0 + raty*( -499.0/1920.0/6.0 + raty*(  59.0/48.0/24.0 + raty*(  59.0/24.0/120.0 + raty*(-5.0/2.0/720.0 -  7.0/5040.0*raty))))));
            asy[7] =   -5.0/2048.0 + raty*(    -75.0/107520.0 + raty*(  259.0/11520.0/2.0 + raty*(   37.0/1920.0/6.0 + raty*(  -7.0/48.0/24.0 + raty*(  -5.0/24.0/120.0 + raty*( 1.0/2.0/720.0 +  1.0/5040.0*raty))))));

            xn -= 3;
            yn -= 3;

            out[i * M + j] = 0.0;

            for (int k = 0; k < 8; ++k)
            {
                ixi = std::max(xn + k, 0);
                ixi = std::min(ixi, nMax);

                for (int m = 0; m < 8; ++m)
                {
                    iyi = std::max(yn + m, 0);
                    iyi = std::min(iyi, mMax);

                    out[i * M + j] += Z[ixi * M + iyi] * asx[k] * asy[m];
                }
            }
        }
    }
}