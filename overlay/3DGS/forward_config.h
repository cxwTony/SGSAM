#ifndef _FORWARD_CONFIG_
#define _FORWARD_CONFIG_

#include <ap_float.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

#include "pre_grouping_config.h"
#include "../utils.h"

using namespace hls;

static constexpr int W = 480;
static constexpr int H = 360;

static const data_t FP_W = 480.0f;
static const data_t FP_H = 360.0f;

static constexpr int LOGN = 8;
static constexpr bool DIRECTION = false;
static constexpr int NUM_RENDERERS = 2;

static const data_t FP_MIN = 0.000001;
static const data_m FP_TMIN = 0.001;
static const data_t FP_0 = 0;
static const data_t FP_1l255 = 1.0f / 255.0f;
static const data_t FP_0o1 = 0.1;
static const data_t FP_0o5 = 0.5;
static const data_m FP_0o99 = 0.99f;
static const data_t FP_1 = 1;
static const data_t FP_0o3 = 0.3;
static const data_t FP_1o3 = 1.3;
static const data_t FP_255 = 255;

data_t computeVecter (
    const data_t3 p,
    const data_t m[16],
    const int idx
);

void forward(    
    const keyval* pre_groups,
    const md_f8_4* md_1,
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    const data_t focal_x, const data_t focal_y,
    const int* accurate_sizes,
    const int num_groups,
    ren pix_data[H][W],
    int &total_count,
    md_n6f2* md_2,
    md_f8_1* md_3,
    md_f8_2* md_4,
    data_t3* out_colors,
    data_t* out_depths,
    data_t* out_ss,
    data_t* final_Ts
);

#endif