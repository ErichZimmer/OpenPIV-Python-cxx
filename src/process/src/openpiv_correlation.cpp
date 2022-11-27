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
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>

// utils
#include "threadpool.hpp"
#include "openpiv_utils.h"

// openpiv
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

    auto correlator = &algos::PocketFFT::cross_correlate_real<core::image, core::g_f>;
    
    core::image<core::g<imgDtype>> output{ (fft.*correlator)( img_a, img_b ) };

    std::vector<imgDtype> cmatrix( img_a.pixel_count() );
    
    for (std::size_t i = 0; i < img_a.pixel_count(); ++i)
        cmatrix[i] = output[i];

    return cmatrix;
};


// Normalozed cross-correlation
std::vector<imgDtype> process_images_standard(
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

    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_b.size(), ia, overlap );

    // align interrrogation windows to 8 imgDtypes
    std::size_t iw = size;
    // if (correlation_method != 0) // commented out for now 
    //     iw = size * 2; // Pad by 2N

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
    auto fft = algos::PocketFFT( paddedWindow );
    auto correlator = &algos::PocketFFT::cross_correlate_real<core::image, core::g_f>;
    auto autocorrelator = &algos::PocketFFT::auto_correlate<core::image, core::g_f>;

    core::image<core::g<imgDtype>> bias_correction{ paddedWindow.width(), paddedWindow.height() };
    core::fill(bias_correction, core::g<imgDtype>{ 1.0 } );
    bias_correction = (fft.*autocorrelator)( bias_correction );
        
    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &img_b,
        &paddedWindow,
        &fft,
        &correlator,
        &bias_correction,
        &vslice
     ]( std::size_t i, const core::rect& ia, 
        core::image<core::g<imgDtype>>& view_a, 
        core::image<core::g<imgDtype>>& view_b,
        core::image<core::g<imgDtype>>& output)
     {
        auto mean_stdA = mean_std(img_a, ia.bottom(), ia.top(), ia.left(), ia.right());
        auto mean_stdB = mean_std(img_b, ia.bottom(), ia.top(), ia.left(), ia.right());

        placeIntoPadded(img_a, view_a, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_stdA[0]);
        placeIntoPadded(img_b, view_b, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_stdB[0]);

        // prepare & correlate
        output = (fft.*correlator)( view_a, view_b );

        /// normalize output and correct bias
        output = output / bias_correction;

        imgDtype sig = mean_stdA[1] * mean_stdB[1];
        for(std::uint32_t j = 0; j < output.pixel_count(); ++j)
            output[j] = output[j] / sig;

        placeIntoCmatrix(cmatrix, output, ia, vslice, i); 
     };

    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        // - split the grid into thread_count chunks
        // - wrap each chunk into a processing for loop and push to thread

        // ensure we don't miss grid locations due to rounding
        std::size_t chunk_size = grid.size() / thread_count;
        std::vector<size_t> chunk_sizes( thread_count, chunk_size );
        chunk_sizes.back() = grid.size() - (thread_count-1)*chunk_size;

        std::size_t i = 0;
        for ( const auto& chunk_size_ : chunk_sizes )
        {
            pool.enqueue(
                [i, chunk_size_, &grid, &processor, &paddedWindow]() {
                    core::gf_image view_a{ paddedWindow.height(), paddedWindow.width() };
                    core::gf_image view_b{ paddedWindow.height(), paddedWindow.width() };
                    core::gf_image output{ paddedWindow.height(), paddedWindow.width() };
                    
                    for ( std::size_t j=i; j<i + chunk_size_; ++j )
                        processor(j, grid[j], view_a, view_b, output);
                } );
            i += chunk_size_;
        }
    }
    else
    {
        core::gf_image view_a{ paddedWindow.height(), paddedWindow.width() };
        core::gf_image view_b{ paddedWindow.height(), paddedWindow.width() };
        core::gf_image output{ paddedWindow.height(), paddedWindow.width() };

        for (std::size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i], view_a, view_b, output);
    }

    return cmatrix;
}


// autocorrelation
/*
std::vector<imgDtype> process_images_auto(
    py::array_t<imgDtype>& np_img_a,
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

    core::gf_image img_a{ convert_image(np_img_a) };
    
    // create a grid for processing
    auto ia = core::size{size, size};
    std::vector<core::rect> grid = core::generate_cartesian_grid( img_a.size(), ia, overlap );

    // padding
    auto paddedWindow = core::size{size, size};
    if (correlation_method != 0)
        paddedWindow = core::size{size * 2, size * 2}; // pad windows by 2N

    // process!
    std::vector<imgDtype> cmatrix(grid.size() * size * size, 0.0);
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
     ]( std::size_t i, const core::rect& ia )
     {
        auto mean_stdA = mean_std(img_a, ia.bottom(), ia.top(), ia.left(), ia.right());

        imgDtype norm = mean_stdA[1] * paddedWindow.area();
        core::gf_image view_a{ ia.width(), ia.height() };

        placeIntoPadded(img_a, view_a, ia.bottom(), ia.top(), ia.left(), ia.right(), mean_stdA[0]);

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
        std::size_t chunk_size = grid.size()/thread_count;
        std::vector<std::size_t> chunk_sizes( thread_count, chunk_size );
        chunk_sizes.back() = grid.size() - (thread_count-1)*chunk_size;

        std::size_t i = 0;
        for ( const auto& chunk_size_ : chunk_sizes )
        {
            pool.enqueue(
                [i, chunk_size_, &grid, &processor]() {
                    for ( std::size_t j=i; j<i + chunk_size_; ++j )
                        processor(j, grid[j]);
                } );
            i += chunk_size_;
        }
    }
    else
    {
        for (std::size_t i = 0; i < grid.size(); ++i)
            processor(i, grid[i]);
    }

    return cmatrix;
};

*/
// error correlation correction (ecc) cross-correlation
// this code does not work currently because it was never ran to begin with
/*
std::vector<imgDtype> process_images_ecc(
    py::array_t<imgDtype>& np_img_a,
    py::array_t<imgDtype>& np_img_b,
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

    core::image<core::g<imgDtype>> img_a{ convert_image(np_img_a) };
    core::image<core::g<imgDtype>> img_b{ convert_image(np_img_b) };
    
    // create a grid for processing
    auto ia = core::size{size, size};
    auto grid = core::generate_cartesian_grid( img_b.size(), ia, overlap );

    // align interrrogation windows to 8 imgDtypes
    std::size_t vlen = 8;
    std::size_t iw = size;
    if (correlation_method != 0)
        iw = size * 2; // Pad by 2N

    // pad window to multiple of vlen
    std::uint32_t fsize = multof(iw, vlen);
    auto paddedWindow = core::size{fsize, fsize};

    // get start and end for padded window slicing
    std::vector<std::uint32_t> vslice(2);
    vslice[0] = static_cast<std::uint32_t>( std::ceil( (fsize - iw) / 2 ) );
    vslice[1] = static_cast<std::uint32_t>( fsize - std::ceil( (fsize - iw) / 2 ) );

    // create output correlation matrix
    std::vector<imgDtype> cmatrix(grid.size() * size * size, 0.0);
    std::uint32_t cmatrix_stride = size * size;

    // get strides for pocketFFT
    std::uint32_t y = paddedWindow.height();
    std::uint32_t x = paddedWindow.width();
    std::uint32_t new_x = x / 2 + 1;

    core::image<core::g<imgDtype>> in_a(x, y);
    core::image<core::g<imgDtype>> in_b(x, y);
    core::image<core::g<imgDtype>> res(x, y);

    core::image<core::complex<imgDtype>> out_a(new_x, y);
    core::image<core::complex<imgDtype>> out_b(new_x, y);
    
    pocketfft::shape_t shape{
        static_cast<std::size_t>(y),
        static_cast<std::size_t>(x)
    };
    pocketfft::shape_t shape2{
        static_cast<std::size_t>(y),
        static_cast<std::size_t>(new_x)
    };

    ptrdiff_t s = sizeof(imgDtype);
    ptrdiff_t s2 = sizeof(core::complex<imgDtype>);

    pocketfft::stride_t stride{ x * s, s };
    pocketfft::stride_t stride2{ new_x * s2, s2 };
    
    pocketfft::shape_t axes = {
        static_cast<std::size_t>(0),
        static_cast<std::size_t>(1)
    };

    // inverse FFT scaling number
    imgDtype scl_fct = 1.0 / (x * y);

    // normalization kernel
    core::image<core::g<imgDtype>> bias_correction{ paddedWindow.width(), paddedWindow.height() };
    core::fill(bias_correction, core::g<imgDtype>{ 1.0 } );

    core::image<core::complex<imgDtype>> bias_correction_cmp{ new_x, y };

    // FFT real to complex 
    pocketfft::r2c<imgDtype>(
        shape, stride, stride2, axes,
        pocketfft::FORWARD, 
        reinterpret_cast<imgDtype*>(&bias_correction[0]), 
        reinterpret_cast<std::complex<imgDtype>*>(&bias_correction_cmp[0]),
        1.0
    );

    // auto correlation power spectrum
    bias_correction_cmp = bias_correction_cmp * core::conj(bias_correction_cmp);

    // inverse FFT complex to real
    pocketfft::c2r<imgDtype>(
        shape, stride2, stride, axes,
        pocketfft::BACKWARD,
        reinterpret_cast<std::complex<imgDtype>*>(&bias_correction_cmp[0]),
        reinterpret_cast<imgDtype*>(&bias_correction[0]),  
        scl_fct
    );

    // Shift correlation matrix so origin is in te center of the matrix
    core::swap_quadrants( bias_correction );

    auto processor = [
        &cmatrix,
        &cmatrix_stride,
        &img_a,
        &img_b,
        &paddedWindow,
        &bias_correction,
        &vslice,
        &shape,
        &shape2,
        &stride,
        &stride2,
        &axes,
        &scl_fct,
        &grid
    ]( std::size_t row, 
        core::image<core::g<imgDtype>>& view_a, 
        core::image<core::g<imgDtype>>& view_b,
        core::image<core::g<imgDtype>>& output,
        core::image<core::complex<imgDtype>>& view_a_comp,
        core::image<core::complex<imgDtype>>& view_b_comp
    ){
        int ind = 0;
        
        ia = grid[ind];
        auto mean_std_A = mean_std(img_a, ia.bottom(), ia.top(), ia.left(), ia.right());
        auto mean_std_B = mean_std(img_b, ia.bottom(), ia.top(), ia.left(), ia.right());

        placeIntoPadded(img_a, view_a, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_std_A[0]);
        placeIntoPadded(img_b, view_b, ia.bottom(), ia.top(), ia.left(), ia.right(), vslice[0], mean_std_B[0]);

        // perform cross correlation
        pocketfft::r2c<imgDtype>(
            shape, stride, stride2, axes,
            pocketfft::FORWARD, 
            reinterpret_cast<imgDtype*>(&view_a[0]), 
            reinterpret_cast<std::complex<imgDtype>*>(&view_a_comp[0]),
            1.0
        );

        pocketfft::r2c<imgDtype>(
            shape, stride, stride2, axes,
            pocketfft::FORWARD, 
            reinterpret_cast<imgDtype*>(&view_b[0]), 
            reinterpret_cast<std::complex<imgDtype>*>(&view_b_comp[0]),
            1.0
        );

        // cross power spectrum
        view_a_comp = view_b_comp * core::conj(view_a_comp);

        // inverse FFT complex to real
        pocketfft::c2r<imgDtype>(
            shape, stride2, stride, axes,
            pocketfft::BACKWARD,
            reinterpret_cast<std::complex<imgDtype>*>(&view_a_comp[0]),
            reinterpret_cast<imgDtype*>(&output[0]),  
            scl_fct
        );

        // fft-shift like MATLAB and scipy.fft.fftshift
        core::swap_quadrants( output );

        // normalize output and correct bias
        output = output / bias_correction;

        imgDtype sig =  mean_std_A[1] * mean_std_B[1];
        for(std::uint32_t j = 0; j < output.pixel_count(); ++j)
            output[j] = output[j] / sig;

        placeIntoCmatrix(cmatrix, output, ia, vslice, i); 
    };

    // When executing the correlation algorithm, we alllocate the arrays here.
    // This is to avoid redundant allocations. 
    // Each thread gets its own allocated arrays. 
    // This could be replaced by using a class with thread_local data.

    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        // - split the grid into thread_count chunks
        // - wrap each chunk into a processing for loop and push to thread

        // ensure we don't miss grid locations due to rounding
        std::size_t chunk_size = grid.size() / thread_count;
        std::vector<size_t> chunk_sizes( thread_count, chunk_size );
        chunk_sizes.back() = grid.size() - (thread_count-1)*chunk_size;

        std::size_t i = 0;
        for ( const auto& chunk_size_ : chunk_sizes )
        {
            pool.enqueue(
                [i, chunk_size_, &grid, &processor, &x, &new_x, &y]() {
                    core::image<core::g<imgDtype>> view_a{ x, y };
                    core::image<core::g<imgDtype>> view_b{ x, y };
                    core::image<core::g<imgDtype>> output{ x, y };
                    core::image<core::complex<imgDtype>> view_a_comp{ new_x, y };
                    core::image<core::complex<imgDtype>> view_b_comp{ new_x, y };

                    for ( std::size_t j=i; j<i + chunk_size_; ++j )
                        processor(
                            j, grid[j], 
                            view_a, view_b, output,
                            view_a_comp, view_b_comp
                        );
                } );
            i += chunk_size_;
        }
    }
    else
    {
        core::image<core::g<imgDtype>> view_a{ x, y };
        core::image<core::g<imgDtype>> view_b{ x, y };
        core::image<core::g<imgDtype>> output{ x, y };
        core::image<core::complex<imgDtype>> view_a_comp{ new_x, y };
        core::image<core::complex<imgDtype>> view_b_comp{ new_x, y };

        for (std::size_t i = 0; i < grid.size(); ++i)
            processor(
                i, grid[i], 
                view_a, view_b, output,
                view_a_comp, view_b_comp
            );
    }

    return cmatrix;
}
*/