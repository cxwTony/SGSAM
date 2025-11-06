#ifndef _UTILS_
#define _UTILS_

#include <ap_float.h>
#include <hls_half.h>
#include <ap_int.h>

using namespace hls;

typedef half data_m;
typedef float data_t;
typedef ap_fixed<32,12> fixed_t;

struct data_t2 {
    data_t x;
    data_t y;
};

struct data_t3 {
    data_t x;
    data_t y;
    data_t z;
};

struct data_t4 {
    data_t x;
    data_t y;
    data_t z;
    data_t w;
};

struct data_m4 {
    data_m x;
    data_m y;
    data_m z;
    data_m w;
};

struct data_m2 {
    data_m x;
    data_m y;
};

struct data_m3 {
    data_m x;
    data_m y;
    data_m z;
};

struct m1b2 {
    data_m last_alpha;
    bool is_depth_zero;
    bool is_color_zero;
};

struct bool4 {
    bool r;
    bool g;
    bool b;
    bool d;
};

struct int4 {
    int x_min;
    int x_max;
    int y_min;
    int y_max;
};

struct md_n6f2 {
    int orig_idx, n_contrib;
    int4 rect;
    data_t2 point_xy;
};

struct md_f8_1 {
    data_t4 coldep;
    data_t4 conic;
};

struct md_f8_2 {
    data_t3 means3D;
    data_t radius;
    data_t3 cov2D;
    data_t w;
};

struct md_n4f10 {
    int4 rect;
    data_t2 point_xy;
    data_t4 coldep;
    data_t4 conic;
};

struct md_n2 {
    int orig_idx, n_contrib;
};

struct md_f8_3 {
    data_t3 dL_dmean3D;
    data_t dL_dradius;
    data_t3 dL_dcolor;
    data_t dL_dopacity;
};

struct md_f8_4 {
    data_t3 means3D;
    data_t radius;
    data_t3 color;
    data_t opacity;
};

struct md_n1f8 {
    int orig_idx;
    md_f8_4 md_temp;
};

struct md_f10 {
    data_m3 dL_dcolor;
    data_m dL_ddepth;
    data_m2 dL_dmean2D;
    data_m3 dL_dconic2D;
    data_m dL_dopacity;
};

struct md_f4_1 {
    data_t3 dL_dcolor;
    data_t dL_dopacity;
};

struct md_f4_2 {
    data_t3 dL_dmean3D;
    data_t dL_dradius;
};

struct md_f6 {
    data_t dL_ddepth;
    data_t2 dL_dmean2D;
    data_t3 dL_dconic2D;
};

struct keyval {
    int key;
    data_t value;
};

struct afr {
    int index;
    data_t4 conic;
    data_t3 color;
    data_t2 point_xy;
    data_t depth;
    int4 rect;
};

struct ren {
    data_m t;
    data_m r;
    data_m g;
    data_m b;
    data_m d;
    data_m accum_r;
    data_m accum_g;
    data_m accum_b;
    data_m accum_d;
};

#endif