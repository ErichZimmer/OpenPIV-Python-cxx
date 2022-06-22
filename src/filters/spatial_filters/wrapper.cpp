#include <cmath>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "constants.h"
#include "kernels.h"
#include "filters.h"

using imgDtype = constants::imgDtype;

// Interface
namespace py = pybind11;

// wrap C++ function with NumPy array IO
#pragma warning(disable: 4244)

py::array_t<imgDtype> intensity_cap_wrapper(
   py::array_t<imgDtype> input,
   imgDtype std_mult = 2.f
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1);
   auto buf2 = result.request();
   
   imgDtype* ptr_out = (imgDtype*) buf2.ptr;
   
   // call pure C++ function
   intensity_cap_filter(
      ptr_out,
      N*M, 
      std_mult
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<imgDtype> intensity_binarize_wrapper(
   py::array_t<imgDtype> input,
   imgDtype threshold = 0.5
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
   auto buf2 = result.request();
   
   imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
   imgDtype* ptr_out = (imgDtype*) buf2.ptr;
   
   // call pure C++ function
   binarize_filter(
      ptr_out,
      ptr_in,
      N*M, 
      threshold
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<imgDtype> low_pass_filter_wrapper(
   py::array_t<imgDtype> input,
   int kernel_size = 3,
   imgDtype sigma = 1
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
   auto buf2 = result.request();
   
   imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
   imgDtype* ptr_out = (imgDtype*) buf2.ptr;
   
   int kernel_type = 0; // gaussian kernel
   auto GKernel = kernels::get_kernel_type(kernel_type)(kernel_size, sigma);

   // call pure C++ function
   apply_kernel_lowpass(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<imgDtype> high_pass_filter_wrapper(
   py::array_t<imgDtype> input,
   int kernel_size = 3,
   imgDtype sigma = 1,
   py::bool_ clip_at_zero = false
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
   auto buf2 = result.request();
   
   imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
   imgDtype* ptr_out = (imgDtype*) buf2.ptr;
   
   int kernel_type = 0; // gaussian kernel
   auto GKernel = kernels::get_kernel_type(kernel_type)(kernel_size, sigma);

   // call pure C++ function
   apply_kernel_highpass(
      ptr_out,
      ptr_in,
      GKernel,
      N, M, 
      kernel_size,
      clip_at_zero
   );
   
   result.resize({N,M});
   
   return result;
}

py::array_t<imgDtype> local_variance_norm_wrapper(
   py::array_t<imgDtype> input,
   int kernel_size = 3,
   imgDtype sigma1 = 2,
   imgDtype sigma2 = 2,
   py::bool_ clip_at_zero = false
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();

   int N = input.shape()[0], M = input.shape()[1];
   
   py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
   auto buf2 = result.request();
   
   py::array_t<imgDtype> temp_buffer = py::array_t<imgDtype>(buf1.size);
   auto buf3 = temp_buffer.request();
   
   imgDtype* ptr_out = (imgDtype*) buf2.ptr;
   imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
   imgDtype* ptr_buf = (imgDtype*) buf3.ptr;
   
   // call pure C++ function
   local_variance_norm(
      ptr_out,
      ptr_in,
      ptr_buf,
      N, M, 
      kernel_size,
      sigma1,
      sigma2,
      clip_at_zero
   );
   
   result.resize({N,M});
   
   return result;
}

void mult_scal(
   imgDtype* output,
   imgDtype* input,
   const int constant,
   int N, int M
){
   int step = M;
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < M; ++j)
      {
         output[step * i + j] = input[step * i + j] * constant;
      }
   }
//   imgDtype test_mean = std::accumulate(std::begin(input), std::end(output), 0.0)/(N*M);
}

py::array_t<imgDtype> test_wrapper(
   py::array_t<imgDtype> input,
   int testConst
){
   // check input dimensions
   if ( input.ndim() != 2 )
      throw std::runtime_error("Input should be 2-D NumPy array");

   auto buf1 = input.request();
   
   int N = input.shape()[0], M = input.shape()[1];

   py::array_t<imgDtype> result = py::array_t<imgDtype>(buf1.size);
   auto buf2 = result.request();
   
   imgDtype* ptr_in  = (imgDtype*) buf1.ptr;
   imgDtype* ptr_out = (imgDtype*) buf2.ptr;

   // call pure C++ function
   mult_scal(
      ptr_out,
      ptr_in,
      N, M, 
      testConst
   );
   
   result.resize({N,M});
   
   return result;
}

#pragma warning(default: 4244)

PYBIND11_MODULE(_spatial_filters,m) {
   m.doc() = "Python interface for filters written in c++.";
   
   m.def("_test_wrapper",
      &test_wrapper, 
      "Test wrapper by multiplying a scalar to an array.",
      py::arg("input"),
      py::arg("testConst") = 5
   );
   m.def("_intensity_cap", 
      &intensity_cap_wrapper,
      "Apply an intensity cap filter to a 2D array",
       py::arg("input"),
       py::arg("std_mult") = 2.f
   );
   m.def("_threshold_binarization", 
      &intensity_binarize_wrapper,
      "Apply an binarization filter to a 2D array",
       py::arg("input"),
       py::arg("threshold") = 0.5f
   );
   m.def("_gaussian_lowpass_filter", 
      &low_pass_filter_wrapper,
      "Apply a gaussian low pass filter to a 2D array",
       py::arg("input"),
       py::arg("kernel_size") = 3, 
       py::arg("sigma") = 1
   );
   m.def("_gaussian_highpass_filter", 
      &high_pass_filter_wrapper,
      "Apply a gaussian high pass filter to a 2D array",
      py::arg("input"),
      py::arg("kernel_size") = 7, 
      py::arg("sigma") = 3,
      py::arg("clip_at_zero") = false
   );
   m.def("_local_variance_normalization", 
      &local_variance_norm_wrapper, 
      "Apply a local variance normalization filter to a 2D array",
      py::arg("input"),
      py::arg("kernel_size") = 7, 
      py::arg("sigma1") = 2,
      py::arg("sigma2") = 2,
      py::arg("clip_at_zero") = false
   );
}
