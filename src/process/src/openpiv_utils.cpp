#include "openpiv_utils.h"

using namespace openpiv;


std::string get_execution_type(int execution_type)
{
    if (execution_type == 0)
        return "pool";
    else
        return "bulk-pool";
}


core::image<core::g<imgDtype>> convert_image(
    const py::array_t<imgDtype, py::array::c_style | py::array::forcecast>& np_img
){
    core::image<core::g<imgDtype>> img(
        static_cast<std::uint32_t>(np_img.shape(1)), 
        static_cast<std::uint32_t>(np_img.shape(0))
    );
    
    std::memcpy(
        img.data(),
        np_img.data(),
        np_img.size()*sizeof(imgDtype)
    );
    
    return img;
}


std::uint32_t multof(
    const std::uint32_t i,
    const std::uint32_t mult
){
    std::uint32_t n = 0;

    while (n < i)
    {
        n = n + mult;
    }

    return n;
}

std::uint32_t nextPower2(
    const std::uint32_t i
){
    std::uint32_t n = 1;

    while (n < i)
    {
        n = n * 2;
    }

    return n;
}


void placeIntoPadded(
    const core::image<core::g<imgDtype>>& image,
    core::image<core::g<imgDtype>>& intWindow,
    int y1,
    int y2,
    int x1,
    int x2,
    std::uint32_t pad,
    imgDtype meanI = 0.0
){
    std::size_t imgY = y1;
    std::size_t imgX = x1;

    std::size_t maxRow = y2 - y1;
    std::size_t maxCol = x2 - x1;

    std::size_t image_stride = image.width();
    std::size_t result_stride = intWindow.width();
    
    imgDtype val = 0.0;

    for (std::size_t row = 0; row < maxRow; ++row)
    {
        for (std::size_t col = 0; col < maxCol; ++col)
        {
            val = image[(imgY + row) * image_stride  + imgX + col] - meanI;
            intWindow[(pad  + row) * result_stride + pad  + col] =  val;
//                ( val > 0.0 ) ? val : 0.0;
        }
    }
}


imgDtype meanI(
    const core::image<core::g<imgDtype>>& img,
    std::size_t y1,
    std::size_t y2,
    std::size_t x1,
    std::size_t x2
){
    imgDtype sum = 0.0;
    
    std::size_t deltaY = (y2 - y1), deltaX = (x2 - x1);
    std::size_t N_M = deltaY * deltaX;

    std::size_t img_stride = img.width();

    for (std::size_t row{y1}; row < y2; ++row)
    {
        for (std::size_t col{x1}; col < x2; ++col)
            sum += img[row * img_stride + col];
    }

    return sum / static_cast<imgDtype>(N_M);
}


std::vector<imgDtype> mean_std(
    const core::image<core::g<imgDtype>>& img,
    std::size_t y1,
    std::size_t y2,
    std::size_t x1,
    std::size_t x2
){
    imgDtype img_sum, img_std_temp = 0.0;
    imgDtype img_mean, img_std = 0.0;
    
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

    img_mean = img_sum / static_cast<imgDtype>(N_M);
    img_std = std::sqrt( (img_std_temp / static_cast<imgDtype>(N_M)) + (img_mean*img_mean) - (2*img_mean*img_mean) );

    std::vector<imgDtype> stat_out(2);
    stat_out[0] = img_mean; 
    stat_out[1] = img_std;

    return stat_out;
}

        
void applyScalarToImage(
    core::image<core::g<imgDtype>>& image,
    imgDtype scalar,
    std::size_t N_M
){
    for (std::size_t i = 0; i < N_M; ++i)
        image[i] = image[i] / scalar;
}


void placeIntoCmatrix(
    std::vector<imgDtype>& cmatrix,
    const core::image<core::g<imgDtype>>& output,
    const core::rect& ia,
    const std::vector<std::uint32_t>& vslice,
    std::size_t ind
){


    std::size_t k = 0;
    std::size_t output_stride = output.width();
    std::size_t window_stride = ia.area();

    for (std::size_t row = vslice[0]; row < vslice[1]; ++row)
    {
        for (std::size_t col = vslice[0]; col < vslice[1]; ++col)
        {
            cmatrix[ind * window_stride + k] = output[row * output_stride + col];
            ++k;
        }
    }
}