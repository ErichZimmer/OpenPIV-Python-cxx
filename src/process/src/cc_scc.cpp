#include "cc_scc.h"

using namespace openpiv;

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
            
            std::cout<<"meanA: " << mean_A << '\n'; // not working.. why?
            
            auto view_a { placeIntoPadded(img_a, paddedWindow, ia, mean_A) };
            auto view_b { placeIntoPadded(img_b, paddedWindow, ia, mean_B) };

            // prepare & correlate
            core::gf_image output{ (fft.*correlator)( view_a, view_b ) };
            placeIntoCmatrix(cmatrix, output, paddedWindow, ia, i);
        }   
     };

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
    
    return cmatrix;
};