#ifndef CC_ALL_H
#define CC_ALL_H

// std
#include <cinttypes>
#include <vector>

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// openpiv
#include "core/image.h"

// utils
#include "constants.h"


namespace py = pybind11;
using namespace openpiv;
using imgDtype = constants::imgDtype;


std::vector<imgDtype> process_window(
    const core::image<core::g<imgDtype>>&,
    const core::image<core::g<imgDtype>>&
);


std::vector<imgDtype> process_images_standard(
    const core::image<core::g<imgDtype>>&,
    const core::image<core::g<imgDtype>>&,
    std::uint32_t,
    std::uint32_t,
    int,
    int
);


/*
std::vector<imgDtype> process_images_autocorrelate(
    py::array_t<imgDtype, py::array::c_style | py::array::forcecast>&,
    std::uint32_t,
    std::uint32_t,
    int,
    int
);
*/

#endif