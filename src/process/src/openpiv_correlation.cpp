// Due to Link Error 2005 on core::generate_cartesian_grid,
// all correlation functions are placed in one file for now.

/*
~~~~~~~~~~~~~~~~~
Table of Contents
~~~~~~~~~~~~~~~~~
1: comments
14: includes
43: auto-correlation
140: standard cross-correlation
232: normalized cross-correlation
*/

#include "openpiv_correlation.h"

// std
#include <atomic>
#include <chrono>
//#include <cinttypes>
#include <fstream>
#include <iostream>
#include <thread>
//#include <vector>
#include <cmath>

// utils
#include "threadpool.hpp"
#include "cc_utils.h"

// openpiv
#include "algos/fft.h"
#include "core/enumerate.h"
#include "core/grid.h"
//#include "core/image.h"
#include "core/image_utils.h"
#include "core/stream_utils.h"
#include "core/vector.h"

using namespace openpiv;


// standard cross-correlation
std::vector<double> process_images_scc(
    core::gf_image& img_a,
    core::gf_image& img_b,
    uint32_t size = 32,
    uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
    // basic setup
    double overlap = 1.0 - (static_cast<double>(overlap_size) / static_cast<double>(size));

    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);

    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_b.size(), ia, overlap );

    // padding
    auto paddedWindow = core::size{size, size};
    if (correlation_method != 0)
        paddedWindow = core::size{size * 2, size * 2}; // pad windows by 2N

    // process!
    std::vector<double> cmatrix(grid.size() * size * size, 0.0);
    uint32_t cmatrix_stride = size * size;

    auto fft = algos::FFT( paddedWindow );
    auto correlator = &algos::FFT::cross_correlate_real<core::image, core::g_f>;

    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &img_b,
        &paddedWindow,
        &fft,
        &correlator
     ]( size_t i, const core::rect& ia )
     {
        if (ia.area() == paddedWindow.area())// no padding, use memcpy
        {
            auto view_a{ core::extract( img_a, ia ) };
            auto view_b{ core::extract( img_b, ia ) };

            // prepare & correlate
            core::gf_image output{ (fft.*correlator)( view_a, view_b ) };
            std::copy(output.begin(), output.end(), cmatrix.begin() + i * cmatrix_stride);
        }
        else // padded, use loops
        {
            double mean_A = meanI(img_a, ia.bottom(), ia.top(), ia.left(), ia.right());
            double mean_B = meanI(img_b, ia.bottom(), ia.top(), ia.left(), ia.right());
                        
            auto view_a { placeIntoPadded(img_a, paddedWindow, ia, mean_A) };
            auto view_b { placeIntoPadded(img_b, paddedWindow, ia, mean_B) };

            // prepare & correlate
            core::gf_image output{ (fft.*correlator)( view_a, view_b ) };
            placeIntoCmatrix(cmatrix, output, paddedWindow, ia, i);
        }   
     };

    if (thread_count > 1)
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
    else
    {
        for (size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i]);
    }

    return cmatrix;
};


// Normalozed cross-correlation
std::vector<double> process_images_ncc(
    core::gf_image& img_a,
    core::gf_image& img_b,
    uint32_t size = 32,
    uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
    // basic setup
    double overlap = 1.0 - (static_cast<double>(overlap_size) / static_cast<double>(size));

    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);

    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_b.size(), ia, overlap );

    // padding
    auto paddedWindow = core::size{size, size};
    if (correlation_method != 0)
        paddedWindow = core::size{size * 2, size * 2}; // pad windows by 2N

    // process!
    std::vector<double> cmatrix(grid.size() * size * size, 0.0);
    uint32_t cmatrix_stride = size * size;

    auto fft = algos::FFT( paddedWindow );
    auto correlator = &algos::FFT::cross_correlate_real<core::image, core::g_f>;

    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &img_b,
        &paddedWindow,
        &fft,
        &correlator
     ]( size_t i, const core::rect& ia )
     {
        auto mean_stdA = mean_std(img_a, ia.bottom(), ia.top(), ia.left(), ia.right());
        auto mean_stdB = mean_std(img_b, ia.bottom(), ia.top(), ia.left(), ia.right());

        double norm = mean_stdA[1] * mean_stdB[1] * paddedWindow.area();

        auto view_a { placeIntoPadded(img_a, paddedWindow, ia, mean_stdA[0]) };
        auto view_b { placeIntoPadded(img_b, paddedWindow, ia, mean_stdB[0]) };

        // prepare & correlate
        core::gf_image output{ (fft.*correlator)( view_a, view_b ) };

        // normalize output
        applyScalarToImage(output, norm, paddedWindow.area());

        placeIntoCmatrix(cmatrix, output, paddedWindow, ia, i);   
     };

    if (thread_count > 1)
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
    else
    {
        for (size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i]);
    }

    return cmatrix;
};


//auto-correlation
std::vector<double> process_images_ncc(
    core::gf_image& img_a,
    uint32_t size = 32,
    uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
    // basic setup
    double overlap = 1.0 - (static_cast<double>(overlap_size) / static_cast<double>(size));

    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);

    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_a.size(), ia, overlap );

    // padding
    auto paddedWindow = core::size{size, size};
    if (correlation_method != 0)
        paddedWindow = core::size{size * 2, size * 2}; // pad windows by 2N

    // process!
    std::vector<double> cmatrix(grid.size() * size * size, 0.0);
    uint32_t cmatrix_stride = size * size;

    auto fft = algos::FFT( paddedWindow );
    auto correlator = &algos::FFT::auto_correlate<core::image, core::g_f>;

    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &paddedWindow,
        &fft,
        &correlator
     ]( size_t i, const core::rect& ia )
     {
        auto mean_stdA = mean_std(img_a, ia.bottom(), ia.top(), ia.left(), ia.right());

        double norm = mean_stdA[1] * paddedWindow.area();

        auto view_a { placeIntoPadded(img_a, paddedWindow, ia, mean_stdA[0]) };

        // prepare & correlate
        core::gf_image output{ (fft.*correlator)( view_a ) };

        // normalize output
        applyScalarToImage(output, norm, paddedWindow.area());

        placeIntoCmatrix(cmatrix, output, paddedWindow, ia, i);   
     };

    if (thread_count > 1)
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
    else
    {
        for (size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i]);
    }

    return cmatrix;
};