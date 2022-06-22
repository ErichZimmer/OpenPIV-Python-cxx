#ifndef VECTOR_BASED_H
#define VECTOR_BASED_H

#include <cmath>
#include <vector>
#include <algorithm>

// check 8 points to see if vector in question has less than 50% of points invalid
void difference_test2D(
    double* u,
    double* v,
    int* mask,
    double threshold,
    uint32_t N,
    uint32_t M
){    
    for (uint32_t i = 1; i < N - 1; ++i)
    {
        // declare variables here in case of future row-based parallelism
        int invalid_flag = 0; 
        double u_of_q, v_of_q, ui, vi;
        
        for (uint32_t j = 1; j < M - 1; ++j)
        {
            // find current u and v vectors
            u_of_q = u[i * N + j];
            v_of_q = v[i * N + j];
            
            // cycle u[i +/- 1, j +/- 1] and v[i +/- 1, j +/- 1] to find invalid points
            for (uint32_t ii = 0; ii < 3; ++ii)
            {
                for (uint32_t jj = 0; jj < 3; ++jj)
                {
                    // check for nans or inf
                    ui = u[(i + ii - 1) * N + (j + jj - 1)];
                    vi = v[(i + ii - 1) * N + (j + jj - 1)];
                    if (std::isfinite(ui) || std::isfinite(vi)) // should nans be treated as invalid points?
                        if ((std::abs( ui - u_of_q ) > threshold) || (std::abs( vi - v_of_q ) > threshold))
                            invalid_flag += 1;
                }
            }
            
            // if more than 4 points are invalid, then mark the vector as invalid
            if (invalid_flag > 4)
                mask[i * N + j] = 1;
                
            // reset invalid flag
            invalid_flag = 0;
        }
    }
}


double median(std::vector<double>& arr)
{
    ssize_t n = arr.size() / 2;
    
    std::nth_element(
        std::begin(arr),
        std::begin(arr) + n,
        std::end(arr)
    );
    
    double arr_n = arr[n];
    
    if (arr.size() % 2 == 1)
        return arr_n;
    else // even kernel size
    {
        double arr_m = *std::max_element(
            std::begin(arr),
            std::begin(arr) + n
        );
        
        return 0.5 * (arr_n + arr[n - 1]);
    }
}



void local_median_test(
    double* u,
    double* v,
    int* mask,
    double threshold,
    uint32_t N,
    uint32_t M,
    uint32_t kernel_radius
){
    int kernel_size = kernel_radius * 2 + 1;
    
    // assume padding is equal to kernel_radius
    for (uint32_t i = kernel_radius; i < N - kernel_radius; ++i)
    {
        // declare variables here in case of future row-based parallelism
        double u_of_q, v_of_q, ui, vi, u_med, v_med, u_off, v_off;
        
        std::vector<double> kernel_u;
        std::vector<double> kernel_v;
        
        for (uint32_t j = kernel_radius; j < M - kernel_radius; ++j)
        { 
            u_of_q = u[i * N + j];
            v_of_q = v[i * N + j];
            
            for (uint32_t ii = 0; ii < kernel_size; ++ii)
            {
                for (uint32_t jj = 0; jj < kernel_size; ++jj)
                {
                    ui = u[(i + ii - kernel_radius) * N + (j + jj - kernel_radius)];
                    vi = v[(i + ii - kernel_radius) * N + (j + jj - kernel_radius)];

                    // only need points around vector
                    if ((std::isfinite(ui) && ((ii != kernel_radius) && (jj != kernel_radius))))
                        kernel_u.push_back(ui);
                    if ((std::isfinite(vi) && ((ii != kernel_radius) && (jj != kernel_radius))))
                        kernel_v.push_back(vi);

                }
            }

            // obtain medians
            u_med = median(kernel_u);
            v_med = median(kernel_v);
            
            if ((std::abs(u_of_q - u_med) > threshold) || 
                (std::abs(v_of_q - v_med) > threshold))
                mask[i * N + j] = 1;
        }
    }
}


#endif