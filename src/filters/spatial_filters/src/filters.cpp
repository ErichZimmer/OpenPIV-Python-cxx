#include <cmath>
#include <vector>
#include <iterator>
#include <functional>
#include <numeric>
#include <cstdint>

#include "kernels.h"
#include "utils.h"

void parallel_bulk( // for future parallel
    std::function<void(std::size_t)>& lambda,
    std::size_t img_rows,
    std::size_t kernel_size,
    std::size_t thread_count = 4
){
    /* Perform bulk processing due to costs of creating/maintaining queues */
    // get chunk size and starting row
    std::size_t chunk_size = img_rows/thread_count, row = kernel_size / 2;
    // allocate vector of shuck sizes
    std::vector<size_t> chunk_sizes( thread_count, chunk_size );
    // fix rounding errors to remove undefined behavior
    chunk_sizes.back() = (img_rows - 2*(kernel_size / 2) - (thread_count-1)*chunk_size);

    for ( const auto& chunk_size_ : chunk_sizes )
    {
        auto processor = [row, chunk_size_, &lambda] ()
        {
            for ( std::size_t j=row; j<row + chunk_size_; ++j )
                lambda(j);
        };
        
        processor(); // would be multi-threaded later...
        row += chunk_size_;
    }
}

void intensity_cap_filter(
    imgDtype* input,
    int N_M,
    imgDtype std_mult = 2.f
){
    imgDtype upper_limit{};

    // calculate mean and std
    auto mean_std{ buffer_mean_std(input, N_M) };

    // calculate cap
    upper_limit = mean_std[0] + std_mult * mean_std[1];

    // perform intensity capping
    buffer_clip(input, 0.f, upper_limit, N_M);
}

void binarize_filter(
    imgDtype* output,
    imgDtype* input,
    int N_M,
    imgDtype threshold
){

    // perform binarization, assuming pixel intensity range of [0..1]
    for (int i{}; i < N_M; ++i)
        output[i] = (input[i] > threshold) ? 1.f : 0.f;
}

void apply_kernel_lowpass(
    imgDtype* output,
    imgDtype* input,
    std::vector<imgDtype>& kernel,
    int img_rows, int img_cols,
    int kernel_size
){
    int step{ img_cols };

    // setup lambda function for column processing
    auto process_row = [
        output,
        input,
        &kernel,
        &img_cols,
        &step,
        &kernel_size
    ]( int _row ) mutable
    {
        for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
            output[step * _row + col] = kernels::apply_conv_kernel(
                input,
                kernel,
                _row, col, step,
                kernel_size
            );
    };

    // process rows serially
    for (int row{ kernel_size / 2 }; row < img_rows - kernel_size / 2; ++row)
        process_row(row);
}


void apply_kernel_highpass(
    imgDtype* output,
    imgDtype* input,
    std::vector<imgDtype>& kernel,
    int img_rows, int img_cols,
    int kernel_size,
    bool clip_at_zero = false
){
    int step{ img_cols };

    // setup lambda function for column processing
    auto process_row = [
        output,
        input,
        &kernel,
        &img_cols,
        &step,
        &kernel_size
    ]( int _row ) mutable
    {
        for (int col{kernel_size / 2}; col < (img_cols - kernel_size / 2); ++col)
            output[step * _row + col] = input[step * _row + col] - kernels::apply_conv_kernel(
                input,
                kernel,
                _row, col, step,
                kernel_size
            );
    };

    // process rows serially
    for (int row{ kernel_size / 2 }; row < img_rows - kernel_size / 2; ++row)
        process_row(row);

    // clip pixel values less than zero if necessary
    if (clip_at_zero) 
        buffer_clip(output, 0.f, 1.f, img_rows * img_cols);
}