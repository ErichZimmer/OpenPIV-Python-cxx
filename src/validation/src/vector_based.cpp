// std
#include <algorithm>

// validation
#include "vector_based.h"

#include <iostream>


// check 8 points to see if vector in question has less than 50% of points invalid
void difference_test2D(
    const double* u,
    const double* v,
    int* mask,
    double threshold_u,
    double threshold_v,
    std::uint32_t N,
    std::uint32_t M
){    
    for (std::uint32_t i = 1; i < N - 1; ++i)
    {
        // declare variables here in case of future row-based parallelism
        int invalid_flag = 0; 
        double u_of_q, v_of_q, ui, vi = 0.0;

        for (std::uint32_t j = 1; j < M - 1; ++j)
        {
            // set invalid flag and mask to zero
            invalid_flag = 0;
            mask[i * M + j] = 0;

            // find current u and v vectors
            u_of_q = u[i * M + j];
            v_of_q = v[i * M + j];

            if ( !std::isfinite(u_of_q) || !std::isfinite(v_of_q) )
                continue; // Don't process nans

            // cycle u[i +/- 1, j +/- 1] and v[i +/- 1, j +/- 1] to find invalid points
            for (std::uint32_t ii = 0; ii < 3; ++ii)
            {
                for (std::uint32_t jj = 0; jj < 3; ++jj)
                {
                    // check for nans or inf
                    ui = u[(i + ii - 1) * M + (j + jj - 1)];
                    vi = v[(i + ii - 1) * M + (j + jj - 1)];
                    if (std::isfinite(ui) || std::isfinite(vi)) // should nans be treated as invalid points?
                    {
                        if ((std::abs( ui - u_of_q ) > threshold_u) || (std::abs( vi - v_of_q ) > threshold_v))
                        {
                            invalid_flag += 1;
                        }
                    }
                }
            }

            // if more than 4 points are invalid, then mark the vector as invalid
            if (invalid_flag > 4)
                mask[i * M + j] = 1;
        }
    }
}


// pass by copy to avoid changing original data
double median(std::vector<double> arr)
{   
    if ( arr.size() == 0 ) // empty kernel
        return 0.0;
    else if ( arr.size() == 1 ) // only one scalar in kernel
        return arr[0];

    std::size_t n = arr.size() / 2;

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

        return 0.5 * (arr_n + arr_m);
    }
}



void local_median_test(
    const double* u,
    const double* v,
    int* mask,
    double threshold_u,
    double threshold_v,
    std::uint32_t N,
    std::uint32_t M,
    std::uint32_t kernel_radius,
    std::size_t kernel_min_size = 0
){
    std::size_t kernel_size = kernel_radius * 2 + 1;
    
    // we only need the 8 vectors around the vector of question (u/v _of_q)
    std::vector<bool> footprint(kernel_size * kernel_size);
    std::size_t half_size = (kernel_size * kernel_size) / 2;
    footprint[half_size] = 1;
    
    // assume padding is equal to kernel_radius
    for (std::uint32_t i = kernel_radius; i < N - kernel_radius; ++i)
    {
        // declare variables here in case of future row-based parallelism
        double u_of_q, v_of_q, ui, vi, u_med, v_med = 0.0;
        
        for (std::uint32_t j = kernel_radius; j < M - kernel_radius; ++j)
        {
            // set mask to 0
            mask[i * M + j] = 0;

            u_of_q = u[i * M + j];
            v_of_q = v[i * M + j];

            if ( !std::isfinite(u_of_q) || !std::isfinite(v_of_q) )
            {
                continue; // Don't process nans
            }

            std::vector<double> kernel_u;
            std::vector<double> kernel_v;

            // cycle u[i +/- 1, j +/- 1] and v[i +/- 1, j +/- 1] to find valid points
            for (std::uint32_t ii = 0; ii < kernel_size; ++ii)
            {
                for (std::uint32_t jj = 0; jj < kernel_size; ++jj)
                {
                    ui = u[(i + ii - kernel_radius) * M + (j + jj - kernel_radius)];
                    vi = v[(i + ii - kernel_radius) * M + (j + jj - kernel_radius)];

                    // only need points around vector
                    if ( std::isfinite(ui) && (footprint[ii * kernel_size + jj] != 1) )
                        kernel_u.push_back(ui);

                    if ( std::isfinite(vi) && (footprint[ii * kernel_size + jj] != 1) )
                        kernel_v.push_back(vi);
                }
            }

            // obtain medians
            u_med = median(kernel_u);
            v_med = median(kernel_v);

            if ( (kernel_u.size() > kernel_min_size) && (kernel_v.size() > kernel_min_size) )
            {
                if ( (std::abs(u_of_q - u_med) > threshold_u) || 
                     (std::abs(v_of_q - v_med) > threshold_v) )
                {
                    mask[i * M + j] = 1;
                }
            }

            else // kernel is too small to be considered valid
            {
                mask[i * M + j] = 1;
            }
        }
    }
}


void normalized_local_median_test(
    const double* u,
    const double* v,
    int* mask,
    double threshold_u,
    double threshold_v,
    std::uint32_t N,
    std::uint32_t M,
    std::uint32_t kernel_radius,
    double eps = 0.1,
    std::size_t kernel_min_size = 0
){
    std::size_t kernel_size = kernel_radius * 2 + 1;
    
    // we only need the 8 vectors around the vector of question (u/v _of_q)
    std::vector<bool> footprint(kernel_size * kernel_size, 1);
    std::size_t half_size = footprint.size() / 2;
    footprint[half_size] = 0;
    
    // assume padding is equal to kernel_radius
    for (std::uint32_t i = kernel_radius; i < N - kernel_radius; ++i)
    {
        // declare variables here in case of future row-based parallelism
        double u_of_q, v_of_q, ui, vi, u_med, v_med, u_res, v_res, u_rm, v_rm = 0.0;

        for (std::uint32_t j = kernel_radius; j < M - kernel_radius; ++j)
        {
            // set mask to 0
            mask[i * M + j] = 0;

            std::vector<double> kernel_u;
            std::vector<double> kernel_v;

            u_of_q = u[i * M + j];
            v_of_q = v[i * M + j];

            if ( !std::isfinite(u_of_q) || !std::isfinite(v_of_q) )
                continue; // Don't process nans

            // cycle u[i +/- 1, j +/- 1] and v[i +/- 1, j +/- 1] to find valid points
            for (std::uint32_t ii = 0; ii < kernel_size; ++ii)
            {
                for (std::uint32_t jj = 0; jj < kernel_size; ++jj)
                {
                    ui = u[(i + ii - kernel_radius) * M + (j + jj - kernel_radius)];
                    vi = v[(i + ii - kernel_radius) * M + (j + jj - kernel_radius)];

                    // only need points around vector
                    if ((std::isfinite(ui) && footprint[ii * kernel_size + jj]))
                        kernel_u.push_back(ui);

                    if ((std::isfinite(vi) && footprint[ii * kernel_size + jj]))
                        kernel_v.push_back(vi);
                }
            }

            // obtain medians
            u_med = median(kernel_u);
            v_med = median(kernel_v);

            // obtain residual
            for (std::size_t ind = 0; ind < kernel_u.size(); ++ind)
                kernel_u[ind] = std::abs(kernel_u[ind] - u_med);
                
            for (std::size_t ind = 0; ind < kernel_v.size(); ++ind)
                kernel_v[ind] = std::abs(kernel_v[ind] - v_med);
            
            u_res = median(kernel_u);
            v_res = median(kernel_v);
            
            // calculate normalized median
            u_rm = std::abs(u_of_q - u_med) / (u_res + eps);
            v_rm = std::abs(v_of_q - v_med) / (v_res + eps);

            if ( (kernel_u.size() > kernel_min_size) && (kernel_v.size() > kernel_min_size) )
            {
                if ( (u_rm > threshold_u) || (v_rm > threshold_v) )
                {
                    mask[i * M + j] = 1;
                }
            }

            else // kernel is too small to be considered valid
            {
                mask[i * M + j] = 1;
            }
        }
    }
}


double test_median(
    const double* arr,
    size_t N_M
){
    double test_scalar = 0.0;
    std::vector<double> arr2sort;
    for (size_t i = 0; i < N_M; ++i)
    {
        test_scalar = arr[i];
        if (std::isfinite(test_scalar))
            arr2sort.push_back(test_scalar);
    }
    return median(arr2sort);
}