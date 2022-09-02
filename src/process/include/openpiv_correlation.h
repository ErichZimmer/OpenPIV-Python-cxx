#ifndef CC_ALL_H
#define CC_ALL_H

// std
#include <cinttypes>
#include <vector>

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::vector<double> process_window(
    py::array_t<double, py::array::c_style | py::array::forcecast>&,
    py::array_t<double, py::array::c_style | py::array::forcecast>&
);


std::vector<double> process_images_standard(
    py::array_t<double, py::array::c_style | py::array::forcecast>&,
    py::array_t<double, py::array::c_style | py::array::forcecast>&,
    std::uint32_t,
    std::uint32_t,
    int,
    int
);


std::vector<double> process_images_autocorrelate(
    py::array_t<double, py::array::c_style | py::array::forcecast>&,
    std::uint32_t,
    std::uint32_t,
    int,
    int
);

#endif