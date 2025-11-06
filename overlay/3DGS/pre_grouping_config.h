#ifndef _PRE_GROUPING_CONFIG_
#define _PRE_GROUPING_CONFIG_

#include <ap_float.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "../utils.h"

using namespace hls;

static constexpr int NUM_COMPARATORS = 256;
static constexpr int N = 256;
static constexpr int SINGLE_COARSE_SIZE = NUM_COMPARATORS * N;
static const data_t MIN_DEPTH = 0.2;

void pre_grouping(
    const data_t3* means3D,
    const data_t* thresholds_in,
    const data_t viewmatrix[16],
    const int size,
    int &num_groups_return,
    keyval* temp_mem,
    keyval* sortouts,
    int* accurate_sizes
);

#endif