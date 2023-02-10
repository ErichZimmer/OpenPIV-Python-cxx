/*
Due to Link Error 2005 on core::generate_cartesian_grid,
all correlation functions are placed in one file for now.
*/

/*
~~~~~~~~~~~~~~~~~
Table of Contents
~~~~~~~~~~~~~~~~~
1:   comments
17:  includes
47:  standard cross-correlation of one interrogation window
72:  standard cross-correlation
178: auto-correlation
*/

#include "openpiv_correlation.h"

// std
#include <atomic>
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>

// utils
#include "threadpool.hpp"
#include "openpiv_utils.h"

// openpiv
#include "algos/fft.h"
#include "algos/pocket_fft.h"
#include "core/enumerate.h"
#include "core/grid.h"
#include "core/image_utils.h"
#include "core/stream_utils.h"
#include "core/vector.h"

using namespace openpiv;


// standard cross-correlation
std::vector<imgDtype> process_window(
    const core::image<core::g<imgDtype>>& img_a,
    const core::image<core::g<imgDtype>>& img_b
){  
    auto fft_size = core::size{ img_a.width(), img_a.height() };

    auto fft = algos::PocketFFT( fft_size );

    auto correlator = &algos::PocketFFT::cross_correlate_real<core::image, core::g<imgDtype>>;
    
    core::image<core::g<imgDtype>> output{ (fft.*correlator)( img_a, img_b ) };

    std::vector<imgDtype> cmatrix( img_a.pixel_count() );
    
    for (std::size_t i = 0; i < img_a.pixel_count(); ++i)
        cmatrix[i] = output[i];

    return cmatrix;
};


// Normalozed cross-correlation
std::vector<imgDtype> images_to_correlation_standard(
    const core::image<core::g<imgDtype>>& img_a,
    const core::image<core::g<imgDtype>>& img_b,
    std::uint32_t size = 32,
    std::uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
    // basic setup
    imgDtype overlap = 1.0 - (static_cast<imgDtype>(overlap_size) / static_cast<imgDtype>(size));

    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);

    // get images
    std::vector<core::image<core::g<imgDtype>>> images;
    
    images.emplace_back(img_a);
    images.emplace_back(img_b);
    
    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( images[0].size(), ia, overlap );

    // zero padding
    std::size_t iw = size;
     if (correlation_method != 0)
         iw = size * 2; // Pad by 2N

    // pad window to multiple of vlen
    std::uint32_t fsize = nextPower2(iw);
    auto paddedWindow = core::size{fsize, fsize};

    // get start and end for padded window slicing
    std::vector<std::uint32_t> vslice(2);
    vslice[0] = static_cast<std::uint32_t>( std::ceil( (fsize - size) / 2 ) );
    vslice[1] = static_cast<std::uint32_t>( fsize - std::ceil( (fsize - size) / 2 ) );

    // create output correlation matrix
    std::vector<imgDtype> cmatrix(grid.size() * size * size, 0.0);
    std::uint32_t cmatrix_stride = size * size;

    // process!
    auto fft = algos::PocketFFT( paddedWindow );
    auto correlator = &algos::PocketFFT::cross_correlate_real<core::image, core::g<imgDtype>>;
    auto autocorrelator = &algos::PocketFFT::auto_correlate<core::image, core::g<imgDtype>>;

    core::image<core::g<imgDtype>> bias_correction{ paddedWindow.width(), paddedWindow.height() };
    core::fill(bias_correction, core::g<imgDtype>{ 1.0 } );
    bias_correction = (fft.*autocorrelator)( bias_correction );
        
    auto processor = [
        &cmatrix,
        &images,
        &paddedWindow,
        &fft,
        &correlator,
        &bias_correction,
        vslice
     ]( std::size_t i, const core::rect& ia )
     {
        core::image<core::g<imgDtype>> view_a{ paddedWindow.width(), paddedWindow.height() };
        core::image<core::g<imgDtype>> view_b{ paddedWindow.width(), paddedWindow.height() };

        auto mean_stdA = mean_std(images[0], ia.bottom(), ia.top(), ia.left(), ia.right());
        auto mean_stdB = mean_std(images[1], ia.bottom(), ia.top(), ia.left(), ia.right());

        placeIntoPadded(images[0], view_a, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_stdA[0]);
        placeIntoPadded(images[1], view_b, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_stdB[0]);

        // correlate
        core::image<core::g<imgDtype>> output{ (fft.*correlator)( view_a, view_b ) };

        // normalize output and correct bias
        output = output / bias_correction;

        imgDtype sig = mean_stdA[1] * mean_stdB[1];
        for(std::uint32_t j = 0; j < output.pixel_count(); ++j)
            output[j] = output[j] / sig;

        placeIntoCmatrix(cmatrix, output, ia, vslice, i); 
    };


    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        size_t i = 0;
        for ( const auto& ia : grid )
        {
            pool.enqueue( [i, ia, &processor](){ processor(i, ia); } );
            ++i;
        }
    }
    else
    {
        for (std::size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i]);
    }
    
    return cmatrix;
}


// autocorrelation
std::vector<imgDtype> images_to_correlation_auto(
    const core::image<core::g<imgDtype>>& img_a,
    std::uint32_t size = 32,
    std::uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
    // basic setup
    imgDtype overlap = 1.0 - (static_cast<imgDtype>(overlap_size) / static_cast<imgDtype>(size));

    std::uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<std::uint32_t>(threads);

    // get images
    std::vector<core::image<core::g<imgDtype>>> images;
    
    images.emplace_back(img_a);
    
    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( images[0].size(), ia, overlap );

    // zero padding
    std::size_t iw = size;
     if (correlation_method != 0)
         iw = size * 2; // Pad by 2N

    // pad window to multiple of vlen
    std::uint32_t fsize = nextPower2(iw);
    auto paddedWindow = core::size{fsize, fsize};

    // get start and end for padded window slicing
    std::vector<std::uint32_t> vslice(2);
    vslice[0] = static_cast<std::uint32_t>( std::ceil( (fsize - iw) / 2 ) );
    vslice[1] = static_cast<std::uint32_t>( fsize - std::ceil( (fsize - iw) / 2 ) );

    // create output correlation matrix
    std::vector<imgDtype> cmatrix(grid.size() * size * size, 0.0);
    std::uint32_t cmatrix_stride = size * size;

    // process!
    auto fft = algos::FFT( paddedWindow );
    auto autocorrelator = &algos::FFT::auto_correlate<core::image, core::g<imgDtype>>;

    core::image<core::g<imgDtype>> bias_correction{ paddedWindow.width(), paddedWindow.height() };
    core::fill(bias_correction, core::g<imgDtype>{ 1.0 } );
    bias_correction = (fft.*autocorrelator)( bias_correction );
        
    auto processor = [
        &cmatrix,
        &images,
        &paddedWindow,
        &fft,
        &autocorrelator,
        &bias_correction,
        vslice
     ]( std::size_t i, const core::rect& ia )
     {
        core::image<core::g<imgDtype>> view_a{ paddedWindow.width(), paddedWindow.height() };

        auto mean_stdA = mean_std(images[0], ia.bottom(), ia.top(), ia.left(), ia.right());

        placeIntoPadded(images[0], view_a, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_stdA[0]);

        // correlate
        core::image<core::g<imgDtype>> output{ (fft.*autocorrelator)( view_a ) };

        // normalize output and correct bias
        output = output / bias_correction;

        imgDtype sig = mean_stdA[1];
        for(std::uint32_t j = 0; j < output.pixel_count(); ++j)
            output[j] = output[j] / sig;

        placeIntoCmatrix(cmatrix, output, ia, vslice, i); 
     };

    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        size_t i = 0;
        for ( const auto& ia : grid )
        {
            pool.enqueue( [i, ia, &processor](){ processor(i, ia); } );
            ++i;
        }
    }
    else
    {
        for (std::size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i]);
    }
    
    return cmatrix;
}


void correlation_based_correction(
    imgDtype* cmatrix_in,
    imgDtype* cmatrix_out,
    std::uint32_t x_size,
    std::uint32_t y_size,
    std::uint32_t x_count,
    std::uint32_t y_count,
    int threads
){
    std::uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<std::uint32_t>(threads);
        
    auto processor = [
        &cmatrix_in,
        &cmatrix_out,
        x_size,
        y_size,
        x_count
    ]( std::uint32_t row )
    {
        core::image<core::g<imgDtype>> old_output{ x_size, y_size };
        core::image<core::g<imgDtype>> output{ x_size, y_size };

        for (std::uint32_t col = 0; col < x_count; ++col)
        {
            std::uint32_t corr_slice = row * x_count + col;
            std::uint32_t corr_index = corr_slice * output.pixel_count();
            
            std::copy_n(
                &cmatrix_in[0] + corr_index,
                output.pixel_count(), 
                output.begin()
            );
            
            if ( col != 0 )
            {
                for (std::uint32_t i = 0; i < output.pixel_count(); ++i)
                {
                    cmatrix_out[corr_index + i] = std::sqrt(std::abs( old_output[i] * output[i] ));
                }
            }
            
            old_output = output;
        }
    };
    
    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        for ( std::size_t ind = 0; ind < y_count; ++ind )
        {
            pool.enqueue( [ind, &processor](){ processor(ind); } );
        }
    }
    else
    {
        for ( std::size_t ind = 0; ind < y_count; ++ind )
        {
            processor(ind);
        }
    }
}

// Normalozed cross-correlation with error correlation correction
std::vector<imgDtype> images_to_correlation_ecc(
    const core::image<core::g<imgDtype>>& img_a,
    const core::image<core::g<imgDtype>>& img_b,
    std::uint32_t size = 32,
    std::uint32_t overlap_size = 16,
    int correlation_method = 0,
    int threads = 0
){
   // basic setup
    imgDtype overlap = 1.0 - (static_cast<imgDtype>(overlap_size) / static_cast<imgDtype>(size));

    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);

    // get images
    std::vector<core::image<core::g<imgDtype>>> images;
    
    images.emplace_back(img_a);
    images.emplace_back(img_b);
    
    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( images[0].size(), ia, overlap );
    auto x_count = 1 + (images[0].width()  - size) / (overlap * size);
    auto y_count = 1 + (images[0].height() - size) / (overlap * size);

    // zero padding
    std::size_t iw = size;
     if (correlation_method != 0)
         iw = size * 2; // Pad by 2N

    // pad window to multiple of vlen
    std::uint32_t fsize = nextPower2(iw);
    auto paddedWindow = core::size{fsize, fsize};

    // get start and end for padded window slicing
    std::vector<std::uint32_t> vslice(2);
    vslice[0] = static_cast<std::uint32_t>( std::ceil( (fsize - size) / 2 ) );
    vslice[1] = static_cast<std::uint32_t>( fsize - std::ceil( (fsize - size) / 2 ) );

    // create output correlation matrix
    std::vector<imgDtype> cmatrix(grid.size() * size * size, 0.0);
    std::uint32_t cmatrix_stride = size * size;

    // process!
    auto fft = algos::PocketFFT( paddedWindow );
    auto correlator = &algos::PocketFFT::cross_correlate_real<core::image, core::g<imgDtype>>;
    auto autocorrelator = &algos::PocketFFT::auto_correlate<core::image, core::g<imgDtype>>;

    core::image<core::g<imgDtype>> bias_correction{ paddedWindow.width(), paddedWindow.height() };
    core::fill(bias_correction, core::g<imgDtype>{ 1.0 } );
    bias_correction = (fft.*autocorrelator)( bias_correction );

    auto processor = [
        &cmatrix,
        &images,
        &paddedWindow,
        &fft,
        &correlator,
        &bias_correction,
        &grid,
        x_count,
        vslice
     ]( std::size_t row)
     {
        core::image<core::g<imgDtype>> old_output{ paddedWindow.width(), paddedWindow.height() };
        core::image<core::g<imgDtype>> new_output{ paddedWindow.width(), paddedWindow.height() };

        for (std::size_t col = 0; col < x_count; ++col)
        {
            core::rect ia = grid[x_count * row + col];

            core::image<core::g<imgDtype>> view_a{ paddedWindow.width(), paddedWindow.height() };
            core::image<core::g<imgDtype>> view_b{ paddedWindow.width(), paddedWindow.height() };

            auto mean_stdA = mean_std(images[0], ia.bottom(), ia.top(), ia.left(), ia.right());
            auto mean_stdB = mean_std(images[1], ia.bottom(), ia.top(), ia.left(), ia.right());

            placeIntoPadded(images[0], view_a, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_stdA[0]);
            placeIntoPadded(images[1], view_b, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_stdB[0]);

            // correlate
            core::image<core::g<imgDtype>> output{ (fft.*correlator)( view_a, view_b ) };

            // normalize output and correct bias
            output = output / bias_correction;

            imgDtype sig = mean_stdA[1] * mean_stdB[1];
            for (std::uint32_t j = 0; j < output.pixel_count(); ++j)
            {
                output[j] = output[j] / sig;
            }
            
            if ( col != 0 )
            {
                for (std::uint32_t j = 0; j < output.pixel_count(); ++j)
                {
                    new_output[j] = output[j] * old_output[j];
                }
            }

            placeIntoCmatrix(cmatrix, new_output, ia, vslice, x_count * row + col); 
            
            old_output = output;
        }
     };

    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        for ( std::size_t ind = 0; ind < y_count; ++ind )
        {
            pool.enqueue( [ind, &processor](){ processor(ind); } );
        }
    }
    else
    {
        for ( std::size_t ind = 0; ind < y_count; ++ind )
        {
            processor(ind);
        }
    }
    
    return cmatrix;
}
