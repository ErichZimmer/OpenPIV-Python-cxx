#ifndef OPENPIV_UTILS_H
#define OPENPIV_UTILS_H

// std
#include <vector>
#include <cinttypes>
#include <cstddef>

// openpiv
#include "core/rect.h"
#include "core/image.h"
#include "core/image_utils.h"
#include "core/vector.h"

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// utils
#include "constants.h"


namespace py = pybind11;
using namespace openpiv;
using imgDtype = constants::imgDtype;


std::string get_execution_type(int);


core::image<core::g<imgDtype>> convert_image(
    const py::array_t<imgDtype, py::array::c_style | py::array::forcecast>&
);


std::uint32_t multof(
    const std::uint32_t,
    const std::uint32_t
);


std::uint32_t nextPower2(
    const std::uint32_t
);


void placeIntoPadded(
    const core::image<core::g<imgDtype>>&,
    core::image<core::g<imgDtype>>&,
    int, int,
    int, int,
    std::uint32_t,
    imgDtype
);


imgDtype meanI(
    const core::image<core::g<imgDtype>>&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

std::vector<imgDtype> mean_std(
    const core::image<core::g<imgDtype>>&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

        
void applyScalarToImage(
    core::image<core::g<imgDtype>>&,
    imgDtype,
    std::size_t
);


void placeIntoCmatrix(
    std::vector<imgDtype>&,
    const core::image<core::g<imgDtype>>&,
    const core::rect&,
    const std::vector<std::uint32_t>&,
    std::size_t
);

#endif