#include "rasterization_config.h"
using namespace hls;

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
) {
    #pragma HLS INTERFACE m_axi port=means3D            offset=slave depth=256   bundle=gmem0
    #pragma HLS INTERFACE m_axi port=thresholds_in      offset=slave depth=256   bundle=gmem1 max_widen_bitwidth=32  max_write_burst_length=1
    #pragma HLS INTERFACE m_axi port=temp_mem           offset=slave depth=256   bundle=gmem2
    #pragma HLS INTERFACE m_axi port=sortouts           offset=slave depth=256   bundle=gmem3
    #pragma HLS INTERFACE m_axi port=accurate_sizes_pre offset=slave depth=256   bundle=gmem4
    #pragma HLS INTERFACE m_axi port=pre_groups         offset=slave depth=256   bundle=gmem5
    #pragma HLS INTERFACE m_axi port=accurate_sizes     offset=slave depth=256   bundle=gmem6
    #pragma HLS INTERFACE m_axi port=md_1_for           offset=slave depth=256   bundle=gmem7
    #pragma HLS INTERFACE m_axi port=md_2_for           offset=slave depth=256   bundle=gmem8
    #pragma HLS INTERFACE m_axi port=md_3_for           offset=slave depth=256   bundle=gmem9
    #pragma HLS INTERFACE m_axi port=md_4_for           offset=slave depth=256   bundle=gmem10
    #pragma HLS INTERFACE m_axi port=md_1_back          offset=slave depth=256   bundle=gmem11
    #pragma HLS INTERFACE m_axi port=md_2_back          offset=slave depth=256   bundle=gmem12
    #pragma HLS INTERFACE m_axi port=md_3_back          offset=slave depth=256   bundle=gmem13
    #pragma HLS INTERFACE m_axi port=md_4_back          offset=slave depth=256   bundle=gmem14
    #pragma HLS INTERFACE m_axi port=dL_dcoldeps        offset=slave depth=48*36 bundle=gmem15 max_widen_bitwidth=32  max_write_burst_length=1
    #pragma HLS INTERFACE m_axi port=out_colors         offset=slave depth=48*36 bundle=gmem16 max_widen_bitwidth=128 max_write_burst_length=1
    #pragma HLS INTERFACE m_axi port=out_depths         offset=slave depth=48*36 bundle=gmem17 max_widen_bitwidth=32  max_write_burst_length=1
    #pragma HLS INTERFACE m_axi port=out_ss             offset=slave depth=48*36 bundle=gmem18 max_widen_bitwidth=32  max_write_burst_length=1
    #pragma HLS INTERFACE m_axi port=final_Ts_for       offset=slave depth=48*36 bundle=gmem19 max_widen_bitwidth=32  max_write_burst_length=1
    #pragma HLS INTERFACE m_axi port=final_Ts_back      offset=slave depth=48*36 bundle=gmem20 max_widen_bitwidth=32  max_write_burst_length=1
    #pragma HLS INTERFACE m_axi port=depth_mask         offset=slave depth=48*36 bundle=gmem21 max_widen_bitwidth=32  max_write_burst_length=1
    #pragma HLS INTERFACE m_axi port=color_mask         offset=slave depth=48*36 bundle=gmem22 max_widen_bitwidth=32  max_write_burst_length=1

    #pragma HLS INTERFACE s_axilite port=means3D            bundle=control
    #pragma HLS INTERFACE s_axilite port=thresholds_in      bundle=control
    #pragma HLS INTERFACE s_axilite port=temp_mem           bundle=control
    #pragma HLS INTERFACE s_axilite port=sortouts           bundle=control
    #pragma HLS INTERFACE s_axilite port=accurate_sizes_pre bundle=control
    #pragma HLS INTERFACE s_axilite port=pre_groups         bundle=control
    #pragma HLS INTERFACE s_axilite port=accurate_sizes     bundle=control
    #pragma HLS INTERFACE s_axilite port=md_1_for           bundle=control
    #pragma HLS INTERFACE s_axilite port=md_2_for           bundle=control
    #pragma HLS INTERFACE s_axilite port=md_3_for           bundle=control
    #pragma HLS INTERFACE s_axilite port=md_4_for           bundle=control
    #pragma HLS INTERFACE s_axilite port=md_1_back          bundle=control
    #pragma HLS INTERFACE s_axilite port=md_2_back          bundle=control
    #pragma HLS INTERFACE s_axilite port=md_3_back          bundle=control
    #pragma HLS INTERFACE s_axilite port=md_4_back          bundle=control
    #pragma HLS INTERFACE s_axilite port=dL_dcoldeps        bundle=control
    #pragma HLS INTERFACE s_axilite port=out_colors         bundle=control
    #pragma HLS INTERFACE s_axilite port=out_depths         bundle=control
    #pragma HLS INTERFACE s_axilite port=out_ss             bundle=control
    #pragma HLS INTERFACE s_axilite port=final_Ts_for       bundle=control
    #pragma HLS INTERFACE s_axilite port=final_Ts_back      bundle=control
    #pragma HLS INTERFACE s_axilite port=depth_mask         bundle=control
    #pragma HLS INTERFACE s_axilite port=color_mask         bundle=control

    #pragma HLS INTERFACE s_axilite port=viewmatrix      bundle=control
    #pragma HLS INTERFACE s_axilite port=projmatrix      bundle=control
    #pragma HLS INTERFACE s_axilite port=focal_x         bundle=control
    #pragma HLS INTERFACE s_axilite port=focal_y         bundle=control
    #pragma HLS INTERFACE s_axilite port=num_groups      bundle=control
    #pragma HLS INTERFACE s_axilite port=size            bundle=control
    #pragma HLS INTERFACE s_axilite port=is_forward      bundle=control
    #pragma HLS INTERFACE s_axilite port=is_pre_grouping bundle=control
    #pragma HLS INTERFACE s_axilite port=return          bundle=control

    ren pix_data[H][W];
    #pragma HLS AGGREGATE variable=pix_data compact=auto
    #pragma HLS BIND_STORAGE variable=pix_data type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=pix_data dim=1 cyclic factor=NUM_RENDERERS
    #pragma HLS ARRAY_PARTITION variable=pix_data dim=2 cyclic factor=NUM_RENDERERS

    if (is_pre_grouping) {
        int num_groups_return = 0;
        pre_grouping(
            means3D,
            thresholds_in,
            viewmatrix,
            size,
            num_groups_return,
            temp_mem,
            sortouts,
            accurate_sizes_pre
        );
        return num_groups_return;
    } else {
        if (is_forward) {
            int num_rendered = 0;
            forward(    
                pre_groups,
                md_1_for,
                viewmatrix,
                projmatrix,
                focal_x, focal_y,
                accurate_sizes,
                num_groups,
                pix_data,
                num_rendered,
                md_2_for,
                md_3_for,
                md_4_for,
                out_colors,
                out_depths,
                out_ss,
                final_Ts_for
            );
            return num_rendered;
        } else {
            backward(
                md_1_back, 
                md_2_back,
                md_3_back,
                dL_dcoldeps,
                final_Ts_back,
                depth_mask,
                color_mask,
                focal_x, focal_y,
                viewmatrix,
                projmatrix,
                size,
                pix_data,
                md_4_back
            );
            return 0;
        }
    }
}