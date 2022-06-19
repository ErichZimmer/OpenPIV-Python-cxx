#ifndef CC_AUTO_H
#define CC_AUTO_H

// std
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <cmath>

// utils
#include "threadpool.hpp"
#include "cc_utils.h"

// openpiv
#include "algos/fft.h"
#include "core/enumerate.h"
#include "core/grid.h"
#include "core/image.h"
#include "core/image_utils.h"
#include "core/stream_utils.h"
#include "core/vector.h"

using namespace openpiv;

std::vector<double> process_images_autocorrelate(
    core::gf_image&,
    uint32_t,
    uint32_t,
    int,
    int
);

#endif