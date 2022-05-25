#ifndef CC_STANDARD_H
#define CC_STANDARD_H

// std
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <cmath>

// utils
#include "threadpool.hpp"
#include "utils.h"

// openpiv
#include "algos/fft.h"
#include "core/enumerate.h"
#include "core/grid.h"
#include "core/image.h"
#include "core/image_utils.h"
#include "core/stream_utils.h"
#include "core/vector.h"

using namespace openpiv;

std::vector<double> process_images_standard(
    core::gf_image& img_a,
    core::gf_image& img_b,
    uint32_t size = 32,
    uint32_t overlap_size = 16,
    int threads = 0,
    int thread_execution = 0
){
    // basic setup
    double overlap = 1.0 - (static_cast<double>(overlap_size) / static_cast<double>(size));
    std::string execution = get_execution_type(thread_execution);
    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    
    if (threads < 1)
        uint32_t thread_count = static_cast<uint32_t>(threads);

    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_b.size(), ia, overlap );

    // process!
    std::vector<double> cmatrix(grid.size() * size * size, 0.0);
    uint32_t cmatrix_stride = size * size;

    auto fft = algos::FFT( ia );
    auto correlator = &algos::FFT::cross_correlate_real<core::image, core::g_f>;

    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &img_b, 
        &fft, 
        &correlator
     ]( size_t i, const core::rect& ia )
     {
        auto view_a{ core::extract( img_a, ia ) };
        auto view_b{ core::extract( img_b, ia ) };

        // prepare & correlate
        core::gf_image output{ (fft.*correlator)( view_a, view_b ) };
        std::copy(output.begin(), output.end(), cmatrix.begin() + i * cmatrix_stride);
     };

    // check execution
    if ( execution == "pool" )
    {
        ThreadPool pool( thread_count );

        size_t i = 0;
        for ( const auto& ia : grid )
        {
            pool.enqueue( [i, ia, &processor](){ processor(i, ia); } );
            ++i;
        }
    }
    else if ( execution == "bulk-pool" )
    {
        ThreadPool pool( thread_count );

        // - split the grid into thread_count chunks
        // - wrap each chunk into a processing for loop and push to thread

        // ensure we don't miss grid locations due to rounding
        size_t chunk_size = grid.size()/thread_count;
        std::vector<size_t> chunk_sizes( thread_count, chunk_size );
        chunk_sizes.back() = grid.size() - (thread_count-1)*chunk_size;


        size_t i = 0;
        for ( const auto& chunk_size_ : chunk_sizes )
        {
            pool.enqueue(
                [i, chunk_size_, &grid, &processor]() {
                    for ( size_t j=i; j<i + chunk_size_; ++j )
                        processor(j, grid[j]);
                } );
            i += chunk_size_;
        }
    }
    return cmatrix;
};

#endif