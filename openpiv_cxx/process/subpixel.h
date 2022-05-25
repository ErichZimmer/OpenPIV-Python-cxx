#ifndef CC_SUBPIXEL_H
#define CC_SUBPIXEL_H

// std
#include <thread>
#include <vector>

// utils
#include "threadpool.hpp"
#include "utils.h"

// openpiv
#include "core/image.h"
#include "core/image_utils.h"
#include "core/vector.h"

void process_cmatrix_2x3(
    double* cmatrix,
    double* results,
    uint32_t maxStep,
    uint32_t stride_2d,
    core::size stride_1d,
    int threads
    
){  
    uint32_t thread_count = std::thread::hardware_concurrency()-1;
    
    if (threads >= 1)
        thread_count = static_cast<uint32_t>(threads);
        
    std::vector<uint32_t> center{ peak_radius, peak_radius };
    
    // allocate sections for results (couldn't pass list of arrays)
    uint32_t U =  maxStep * 0;
    uint32_t V = maxStep * 1;
    uint32_t PH = maxStep * 2;
    uint32_t P2P = maxStep * 3;
    
    auto processor = [
        cmatrix,
        results,
        &maxStep,
        &stride_2d, 
        &stride_1d, 
        &U, &V, &PH, &P2P,
        &num_peaks,
        &peak_radius,
        &center
     ]( uint32_t step )
    {
        auto corrCut = core::gf_image(stride_1d);
        std::copy(
            cmatrix + (step  * stride_2d),
            cmatrix + ((step + 1) * stride_2d),
            corrCut.begin()
        );
        // find peaks
        constexpr uint16_t num_peaks = 2;
        constexpr uint16_t radius = 1;
        
        core::peaks_t<core::g_f> peaks = core::find_peaks( corrCut, num_peaks, peak_radius );
        
        // sub-pixel fitting
        if ( peaks.size() != num_peaks )
        {
            results[step + U] = NAN;
            results[step + V] = NAN;
            results[step + PH] = NAN;
            results[step + P2P] = NAN;
            return;
        }
            
        core::point2<double> uv;
        uv = core::fit_simple_gaussian( peaks[0] );
        
        results[step + U]  = uv[0] - stride_1d.width()/2;
        results[step + V]  = uv[1] - stride_1d.height()/2;
        results[step + PH] = peaks[0][ {center[0], center[1]} ];

        if ( peaks[1][ {center[0], center[1]} ] > 0 )
            results[step + P2P] = peaks[0][ {center[0], center[1]} ] / peaks[1][ {center[0], center[1]} ];
    };
    
    ThreadPool pool( thread_count );

    // - split the grid into thread_count chunks
    // - wrap each chunk into a processing for loop and push to thread

    // ensure we don't miss grid locations due to rounding
    size_t chunk_size = maxStep/thread_count;
    std::vector<size_t> chunk_sizes( thread_count, chunk_size );
    chunk_sizes.back() = maxStep - (thread_count-1)*chunk_size;


    size_t i = 0;
    for ( const auto& chunk_size_ : chunk_sizes )
    {
        pool.enqueue(
            [i, chunk_size_, &processor]() {
                for ( size_t j=i; j<i + chunk_size_; ++j )
                    processor(j);
            } );
        i += chunk_size_;
    }
}

#endif