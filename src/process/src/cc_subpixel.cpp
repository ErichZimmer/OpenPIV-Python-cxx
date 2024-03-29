#include "cc_subpixel.h"

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

// openpiv
//#include "core/image.h"
//#include "core/image_utils.h"
//#include "core/vector.h"

using namespace openpiv;

/*
struct is_peak_struct{
    std::uint32_t h = 0;
    std::uint32_t w = 0;
    core::g_f val = 0.0;
    bool ispeak = false;
};


core::peaks_t<core::g_f> find_peaks_brute(
    const core::gf_image& im, 
    std::uint16_t num_peaks,
    std::uint32_t peak_radius
){
    core::peaks_t<core::g_f> result;
    const std::uint32_t result_w = 2*peak_radius + 1;
    const std::uint32_t result_h = result_w;

    // create vector of peak indexes    
    auto bl = im.rect().bottomLeft();

    core::g_f previous_max = 9999999999.99;
    is_peak_struct temp_peak;
    temp_peak.ispeak = false;

    for (auto i=num_peaks; i>0; --i)
    {
        for ( std::uint32_t h=peak_radius; h<im.height()-2*peak_radius; ++h )
        {
            const core::g_f* above = im.line( h-1 );
            const core::g_f* line = im.line( h );
            const core::g_f* below = im.line( h+1 );

            for ( std::uint32_t w=peak_radius; w<im.width()-peak_radius; ++w )
            {
                if ( line[w-1] < line[w] && line[w+1] < line[w] && above[w] < line[w]  && below[w] < line[w] && // is local peak?
                     line[w] > temp_peak.val && line[w] < previous_max) // is correct local peak?
                {
                    temp_peak.h = h;
                    temp_peak.w = w;
                    temp_peak.val = line[w];
                    temp_peak.ispeak = true;
                }
            }
        }

        if (!temp_peak.ispeak)
            break;
        else
        {
            result.emplace_back(
                core::extract(
                    im,
                    core::rect( {bl[0] + temp_peak.w - peak_radius, bl[1] + temp_peak.h - peak_radius}, {result_w, result_h} )
                )
            );

            previous_max = temp_peak.val;
            temp_peak.val = 0.0;
            temp_peak.ispeak = false;
        }
    }
    // result.resize(num_peaks);
    return result;
}
*/


void process_cmatrix_2x3(
    double* cmatrix,
    double* results,
    std::uint32_t maxStep,
    std::uint32_t stride_2d,
    std::vector<std::uint32_t> stride_1d,
    int limit_peak_search,
    int threads,
    int return_type
    
){
    std::uint32_t thread_count = std::thread::hardware_concurrency()-1;

    if (threads >= 1)
        thread_count = static_cast<std::uint32_t>(threads);

    // allocate sections for results (couldn't pass list of arrays)
    std::size_t U   = maxStep * 0;
    std::size_t V   = maxStep * 1;
    std::size_t PH  = maxStep * 2;
    std::size_t P2P = maxStep * 3;
    std::size_t U2  = maxStep * 4;
    std::size_t V2  = maxStep * 5;
    std::size_t U3  = maxStep * 6;
    std::size_t V3  = maxStep * 7;

    auto processor = [
        cmatrix,
        results,
        stride_2d,
        &stride_1d,
        &U, &V, &PH, &P2P,
        &U2, &V2,
        &U3, &V3,
        return_type
     ]( std::size_t step )
     {
        uint16_t num_peaks = 3;
        constexpr uint16_t radius = 1;

        auto ia = core::size{stride_1d[0], stride_1d[1]};

        auto corrCut = core::gf_image(ia);

        std::copy(
            cmatrix + (step  * stride_2d),
            cmatrix + ((step + 1) * stride_2d),
            corrCut.begin()
        );

        // find peaks
        core::peaks_t<core::g_f> peaks = core::find_peaks( corrCut, num_peaks, radius );

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
        if (return_type == 1 || return_type == 0) // peak 1
        {
            uv = core::fit_simple_gaussian( peaks[0] );
            results[step + U] = uv[0] - static_cast<double>( stride_1d[0]/2 );
            results[step + V] = uv[1] - static_cast<double>( stride_1d[1]/2 );
        }

        if (return_type == 2 || return_type == 0) // peak 2
        {
            uv = core::fit_simple_gaussian( peaks[1] );
            results[step + U2] = uv[0] - static_cast<double>( stride_1d[0]/2 );
            results[step + V2] = uv[1] - static_cast<double>( stride_1d[1]/2 );
        }

        if (return_type == 3 || return_type == 0) // peak 3
        {
            uv = core::fit_simple_gaussian( peaks[2] );
            results[step + U3] = uv[0] - static_cast<double>( stride_1d[0]/2 );
            results[step + V3] = uv[1] - static_cast<double>( stride_1d[1]/2 );
        }

        // primary peak information
        results[step + PH] = peaks[0][ {radius, radius} ];

        if ( peaks[1][ {radius, radius} ] > 0 )
            results[step + P2P] = peaks[0][ {radius, radius} ] / peaks[1][ {radius, radius} ];

    };

    if (thread_count > 1)
    {
        ThreadPool pool( thread_count );

        // - split the grid into thread_count chunks
        // - wrap each chunk into a processing for loop and push to thread

        // ensure we don't miss grid locations due to rounding
        std::size_t chunk_size = maxStep/thread_count;
        std::vector<std::size_t> chunk_sizes( thread_count, chunk_size );
        chunk_sizes.back() = maxStep - (thread_count-1)*chunk_size;


        std::size_t i = 0;
        for ( const auto& chunk_size_ : chunk_sizes )
        {
            pool.enqueue(
                [i, chunk_size_, &processor]() {
                    for ( std::size_t j = i; j < i + chunk_size_; ++j )
                        processor(j);
                } );
            i += chunk_size_;
        }
    } 
    else 
    {
        for ( std::size_t i = 0; i < maxStep; ++i )
            processor(i);
    }
        
}