#ifndef _BACKWARD_CONFIG_
#define _BACKWARD_CONFIG_

#include <ap_float.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

#include "../utils.h"
#include "forward_config.h"

using namespace hls;

static constexpr data_t DDELX_DX = 0.5 * W;
static constexpr data_t DDELY_DY = 0.5 * H;

static const data_m color_mul = data_m(1.0f);
static const data_m depth_mul = data_m(6.0f);
static constexpr data_t d_div = 1.0f / data_t(6.0f * W * H);
static constexpr int NUM_ADDERS = 8;

static const data_m HFP_256 = 16.0f;
static constexpr data_t d_div_plus = 1.0f / data_t(6.0f * W * H * 16.0f);

static const data_m HFP_0 = 0.0f;
static const data_m HFP_1 = 1.0f;
static const data_m HFP_0o5 = 0.5f;
static const data_m HFP_0o99 = 0.99f;

void backward(
    const md_n6f2* md_1, 
    const md_f8_1* md_2,
    const md_f8_2* md_3,
    const bool4* dL_dcoldeps,
    const data_t* final_Ts,
    const bool* depth_mask,
    const bool* color_mask,
    const data_t focal_x, data_t focal_y,
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    const int size,
    ren pix_data[H][W],
    md_f8_3* md_4
);

#endif