#include "pre_grouping_config.h"
using namespace hls;

void thresholds_initial(data_t thresholds[NUM_COMPARATORS], int coarse_count[NUM_COMPARATORS], const data_t* thresholds_in) {
    #pragma HLS INLINE off
    for (int i = 0; i < NUM_COMPARATORS; ++i) {
        #pragma HLS PIPELINE
        thresholds[i] = thresholds_in[i];
        coarse_count[i] = 0;
    }
}

void computeDepth(
    const data_t3* means3D,
    const data_t m[16],
    const int size,
    stream<data_t> &depth_out
) {
    #pragma HLS INLINE off
    computeDepth_LOOP: for (int idx = 0; idx < size; ++idx) {
        #pragma HLS PIPELINE II=8

        const data_t3 p = means3D[idx];
        const data_t x = m[2] * p.x; 
        const data_t y = m[6] * p.y; 
        const data_t z = m[10] * p.z; 
        const data_t temp1 = x + y;
        const data_t temp2 = z + m[14];
        const data_t depth = temp1 + temp2;
        depth_out.write(depth);
    }
}

void compare_unit(
    const data_t depth,
    const data_t thresholds[NUM_COMPARATORS],
    int &group_id
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=thresholds cyclic factor=NUM_COMPARATORS/8

    for (int i = 0; i < NUM_COMPARATORS; ++i) {
        #pragma HLS UNROLL factor=NUM_COMPARATORS/8
        const bool temp = (depth >= thresholds[i]);
        group_id += temp;
    }
}

void coarseCompare(
    stream<data_t> &depth_in,
    const data_t thresholds[NUM_COMPARATORS],
    const int size,
    keyval* temp_mem,
    int coarse_count[NUM_COMPARATORS]
) {
    #pragma HLS INLINE off

    coarse_LOOP: for (int idx = 0; idx < size; ++idx) {
        #pragma HLS PIPELINE II=8

        const data_t depth = depth_in.read();
        int group_id = 0;
        compare_unit(
            depth,
            thresholds,
            group_id
        );

        if (depth > MIN_DEPTH && group_id < NUM_COMPARATORS) {
            const int current_count = coarse_count[group_id];
    
            const int temp_addr = (group_id * SINGLE_COARSE_SIZE) + current_count;
            temp_mem[temp_addr] = {idx, depth};
            coarse_count[group_id] = current_count + 1;
        }
    }
}

int initAccurate(
    const int idx,
    const data_t thresholds[NUM_COMPARATORS],
    data_t accurate_thresholds[NUM_COMPARATORS]
) {
    #pragma HLS INLINE
    const data_t temp2 = (idx == 0) ? MIN_DEPTH : thresholds[idx - 1];
    const int num_thres = NUM_COMPARATORS;  
    const data_t temp5 = thresholds[idx] - temp2;
    const data_t step = temp5 / num_thres;

    for (int i = 0; i < NUM_COMPARATORS; ++i) {
        #pragma HLS PIPELINE II=1
        const data_t temp3 = step * (i + 1);
        accurate_thresholds[i] = temp2 + temp3;
    }

    return num_thres;
}

void computeGroup(
    const int idx,
    const keyval* temp_mem,
    const int group_size,
    const int num_thres,
    const data_t accurate_thresholds[NUM_COMPARATORS],
    int accurate_count[NUM_COMPARATORS],
    int &num_groups,
    keyval* sortouts,
    int* accurate_sizes
) {
    #pragma HLS INLINE

    for (int i = 0; i < group_size; ++i) {
        #pragma HLS PIPELINE II=8
        const int temp_idx1 = idx * SINGLE_COARSE_SIZE + i;
        const keyval temp = temp_mem[temp_idx1];
        int group_id = 0;
        compare_unit(
            temp.value,
            accurate_thresholds,
            group_id
        );

        int cnt = accurate_count[group_id];
        if (group_id < NUM_COMPARATORS && cnt < N) {
            const int temp_idx2 = (num_groups + group_id) * N;
            const int temp_idx3 = temp_idx2 + cnt;
            sortouts[temp_idx3] = temp;
            cnt++;
            accurate_count[group_id] = cnt;
        }
    }

    for (int i = 0; i < num_thres; ++i) {
        #pragma HLS PIPELINE II=1
        const int temp_idx4 = num_groups + i;
        accurate_sizes[temp_idx4] = accurate_count[i];
    }
    num_groups += num_thres;
}

void accurateCompare(
    const keyval* temp_mem,
    const int coarse_count[NUM_COMPARATORS],
    const data_t thresholds[NUM_COMPARATORS],
    int &num_groups,
    keyval* sortouts,
    int* accurate_sizes
) {
    #pragma HLS INLINE off
    accurate_LOOP: for (int idx = 0; idx < NUM_COMPARATORS; ++idx) {
        const int group_size = coarse_count[idx];
        if (group_size == 0)
            continue;

        data_t accurate_thresholds[NUM_COMPARATORS];
        int accurate_count[NUM_COMPARATORS] = {0};
        const int num_thres = initAccurate(idx, thresholds, accurate_thresholds);
    
        computeGroup(idx, temp_mem, group_size, num_thres, accurate_thresholds, accurate_count, num_groups, sortouts, accurate_sizes);
    }
}

void pre_grouping_dataflow(
    const data_t3* means3D,
    const data_t thresholds[NUM_COMPARATORS],
    const data_t viewmatrix[16],
    const int size,
    int coarse_count[NUM_COMPARATORS],
    keyval* temp_mem
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off
    
    stream<data_t, 64> stream_depth;

    computeDepth(
        means3D,
        viewmatrix,
        size,
        stream_depth
    );

    coarseCompare(
        stream_depth,
        thresholds,
        size,
        temp_mem,
        coarse_count
    );
}

void pre_grouping(
    const data_t3* means3D,
    const data_t* thresholds_in,
    const data_t viewmatrix[16],
    const int size,
    int &num_groups_return,
    keyval* temp_mem,
    keyval* sortouts,
    int* accurate_sizes
) { 
    #pragma HLS INLINE off

    int coarse_count[NUM_COMPARATORS];
    data_t thresholds[NUM_COMPARATORS];

    thresholds_initial(thresholds, coarse_count, thresholds_in);

    pre_grouping_dataflow(
        means3D,
        thresholds,
        viewmatrix,
        size,
        coarse_count,
        temp_mem
    );

    accurateCompare(
        temp_mem,
        coarse_count,
        thresholds,
        num_groups_return,
        sortouts,
        accurate_sizes
    );
}