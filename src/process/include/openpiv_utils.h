#ifndef OPENPIV_UTILS_H
#define OPENPIV_UTILS_H

// std
#include <vector>
// #include <cinttypes>
#include <cstddef>

// openpiv
#include "core/rect.h"
#include "core/image.h"
#include "core/image_utils.h"
#include "core/vector.h"

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace openpiv;
namespace py = pybind11;


std::string get_execution_type(int);


core::gf_image convert_image(
    py::array_t<double, py::array::c_style | py::array::forcecast>&
);


void placeIntoPadded(
    const core::gf_image&,
    core::gf_image&,
    int, int,
    int, int,
    double
);


double meanI(
    const core::gf_image&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

std::vector<double> mean_std(
    const core::gf_image&,
    std::size_t,
    std::size_t,
    std::size_t,
    std::size_t
);

        
void applyScalarToImage(
    core::gf_image&,
    double,
    std::size_t
);


void placeIntoCmatrix(
    std::vector<double>&,
    const core::gf_image&,
    const core::size&,
    const core::rect&,
    std::size_t
);

#endif