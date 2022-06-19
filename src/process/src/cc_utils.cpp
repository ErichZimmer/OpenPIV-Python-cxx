#include "cc_utils.h"

using namespace openpiv;


std::string get_execution_type(int execution_type)
{
    if (execution_type == 0)
        return "pool";
    else
        return "bulk-pool";
}


core::gf_image placeIntoPadded(
    core::gf_image& image,
    core::size& padSize,
    const core::rect& ia,
    double meanI = 0
){
    core::gf_image result{ padSize.height(), padSize.width()}; 

    const size_t padY = padSize.height() / 2 - ia.height() / 2;
    const size_t padX = padSize.width() / 2 - ia.width() / 2;

    ssize_t imgY = ia.bottom();
    ssize_t imgX = ia.left();

    ssize_t maxRow = ia.height();
    ssize_t maxCol = ia.width();

    ssize_t image_stride = image.width();
    ssize_t result_stride = result.width();

    for (size_t row = 0; row < maxRow; ++row)
    {
        for (size_t col = 0; col < maxCol; ++col)
            result[(padY + row) * result_stride + padX + col] = image[(imgY + row) * image_stride + imgX + col] - meanI;
    }    
    return result;
}


double meanI(
    core::gf_image& img,
    std::size_t y1,
    std::size_t y2,
    std::size_t x1,
    std::size_t x2
){
    double sum{}, mean{}, std_{};
    
    size_t deltaY = (y2 - y1), deltaX = (x2 - x1);
    size_t N_M = deltaY * deltaX;

    size_t img_stride = img.width();

    for (size_t row{y1}; row < y2; ++row)
    {
        for (size_t col{x1}; col < x2; ++col)
            sum += img[row * img_stride + col];
    }

    return sum / N_M;
}


std::vector<double> mean_std(
    core::gf_image& img,
    std::size_t y1,
    std::size_t y2,
    std::size_t x1,
    std::size_t x2
){
    double img_sum{}, img_mean{}, img_std{}, img_std_temp{};

    size_t deltaY = (y2 - y1), deltaX = (x2 - x1);
    size_t N_M = deltaY * deltaX;
    size_t img_stride = img.width();

    for (size_t row{y1}; row < y2; ++row)
    {
        for (size_t col{x1}; col < x2; ++col)
        {
            img_sum += img[row * img_stride + col];
            img_std_temp += img[row * img_stride + col]*img[row * img_stride + col];
        }
    }

    img_mean = img_sum / N_M;
    img_std = sqrt( (img_std_temp / N_M) + (img_mean*img_mean) - (2*img_mean*img_mean) );

    std::vector<double> stat_out(2);
    stat_out[0] = img_mean; 
    stat_out[1] = img_std;

    return stat_out;
}

        
void applyScalarToImage(
    core::gf_image& image,
    double scalar,
    ssize_t N_M
){
    for (ssize_t i = 0; i < N_M; ++i)
        image[i] = image[i] / scalar;
}


void placeIntoCmatrix(
    std::vector<double>& cmatrix,
    core::gf_image output,
    core::size padSize,
    core::rect ia,
    uint32_t ind
){
    const size_t padY = padSize.height() / 2 - ia.height() / 2;
    const size_t padX = padSize.width() / 2 - ia.width() / 2;

    size_t k = 0;
    ssize_t output_stride = output.width();
    ssize_t window_stride = ia.area();

    for (size_t row = { padY }; row < padSize.height() - padY; ++row)
    {
        for (size_t col = { padX }; col < padSize.width() - padX; ++col)
        {
            cmatrix[ind * window_stride + k] = output[row * output_stride + col];
            ++k;
        }
    }
    /*
    std::copy(
        output.begin() + row + padX, 
        output.begin() + row + padX + ia.width(), 
        cmatrix.begin() + ind * (ia.area())
    );
    */
}