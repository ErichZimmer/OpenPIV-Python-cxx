#include "openpiv_utils.h"

using namespace openpiv;


std::string get_execution_type(int execution_type)
{
    if (execution_type == 0)
        return "pool";
    else
        return "bulk-pool";
}


core::gf_image convert_image(
    py::array_t<double, py::array::c_style | py::array::forcecast>& np_img
){
    core::gf_image img(
        static_cast<std::uint32_t>(np_img.shape(1)), 
        static_cast<std::uint32_t>(np_img.shape(0))
    );
    
    std::memcpy(
        img.data(),
        np_img.data(),
        np_img.size()*sizeof(double)
    );
    
    return img;
}


void placeIntoPadded(
    const core::gf_image& image,
    core::gf_image& intWindow,
    int y1,
    int y2,
    int x1,
    int x2,
    double meanI = 0.0
){
    const std::size_t padY = intWindow.height() / 2 - (y2 - y1) / 2;
    const std::size_t padX = intWindow.width()  / 2 - (x2 - x1) / 2;

    std::size_t imgY = y1;
    std::size_t imgX = x1;

    std::size_t maxRow = y2 - y1;
    std::size_t maxCol = x2 - x1;

    std::size_t image_stride = image.width();
    std::size_t result_stride = intWindow.width();

    for (std::size_t row = 0; row < maxRow; ++row)
        for (std::size_t col = 0; col < maxCol; ++col)
            intWindow[(padY + row) * result_stride + padX + col] = 
                image[(imgY + row) * image_stride  + imgX + col] - meanI;
}


double meanI(
    const core::gf_image& img,
    std::size_t y1,
    std::size_t y2,
    std::size_t x1,
    std::size_t x2
){
    double sum, std_ = 0.0;
    
    std::size_t deltaY = (y2 - y1), deltaX = (x2 - x1);
    std::size_t N_M = deltaY * deltaX;

    std::size_t img_stride = img.width();

    for (std::size_t row{y1}; row < y2; ++row)
    {
        for (std::size_t col{x1}; col < x2; ++col)
            sum += img[row * img_stride + col];
    }

    return sum / static_cast<double>(N_M);
}


std::vector<double> mean_std(
    const core::gf_image& img,
    std::size_t y1,
    std::size_t y2,
    std::size_t x1,
    std::size_t x2
){
    double img_sum, img_std_temp = 0.0;
    double img_mean, img_std = 0.0;
    
    std::size_t deltaY = (y2 - y1), deltaX = (x2 - x1);
    std::size_t N_M = deltaY * deltaX;
    std::size_t img_stride = img.width();

    for (std::size_t row{y1}; row < y2; ++row)
    {
        for (std::size_t col{x1}; col < x2; ++col)
        {
            img_sum += img[row * img_stride + col];
            img_std_temp += img[row * img_stride + col]*img[row * img_stride + col];
        }
    }

    img_mean = img_sum / static_cast<double>(N_M);
    img_std = std::sqrt( (img_std_temp / static_cast<double>(N_M)) + (img_mean*img_mean) - (2*img_mean*img_mean) );

    std::vector<double> stat_out(2);
    stat_out[0] = img_mean; 
    stat_out[1] = img_std;

    return stat_out;
}

        
void applyScalarToImage(
    core::gf_image& image,
    double scalar,
    std::size_t N_M
){
    for (std::size_t i = 0; i < N_M; ++i)
        image[i] = image[i] / scalar;
}


void placeIntoCmatrix(
    std::vector<double>& cmatrix,
    const core::gf_image& output,
    const core::size& padSize,
    const core::rect& ia,
    std::size_t ind
){
    const std::size_t padY = padSize.height() / 2 - ia.height() / 2;
    const std::size_t padX = padSize.width() / 2 - ia.width() / 2;

    std::size_t k = 0;
    std::size_t output_stride = output.width();
    std::size_t window_stride = ia.area();

    for (std::size_t row = { padY }; row < padSize.height() - padY; ++row)
    {
        for (std::size_t col = { padX }; col < padSize.width() - padX; ++col)
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