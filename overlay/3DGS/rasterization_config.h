#ifndef _RASTERIZATION_CONFIG_
#define _RASTERIZATION_CONFIG_

#include "pre_grouping_config.h"
#include "forward_config.h"
#include "backward_config.h"

int rasterization(
// pre_grouping
    const data_t3* means3D,
    const data_t* thresholds_in,
    keyval* temp_mem,
    keyval* sortouts,
    int* accurate_sizes_pre,
// forward
    const keyval* pre_groups,
    const md_f8_4* md_1_for,
    const int* accurate_sizes,
    md_n6f2* md_2_for,
    md_f8_1* md_3_for,
    md_f8_2* md_4_for,
    data_t3* out_colors,
    data_t* out_depths,
    data_t* out_ss,
    data_t* final_Ts_for,
// backward
    const md_n6f2* md_1_back, 
    const md_f8_1* md_2_back,
    const md_f8_2* md_3_back,
    const bool4* dL_dcoldeps,
    const data_t* final_Ts_back,
    const bool* depth_mask,
    const bool* color_mask,
    md_f8_3* md_4_back,
// public
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    const data_t focal_x, const data_t focal_y,
    const int num_groups,
    const int size,
    const bool is_forward,
    const bool is_pre_grouping
);

#endif