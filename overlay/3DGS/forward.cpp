#include "forward_config.h"
#include <hls_math.h>
using namespace hls;

data_t computeVecter (
    const data_t3 p,
    const data_t m[16],
    const int idx
) {
    #pragma HLS INLINE

    const data_t temp1 = m[8 + idx] * p.z + m[12 + idx];
    const data_t temp2 = m[4 + idx] * p.y + temp1;
    const data_t sum = m[0 + idx] * p.x + temp2;
    
    return sum;
}

void merge_unit(
    const keyval src[N], 
    const int left, 
    const int mid, 
    const int right, 
    keyval dst[N]
) {
    #pragma HLS INLINE
    #pragma HLS DEPENDENCE variable=src inter false
    #pragma HLS DEPENDENCE variable=dst inter false

    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        #pragma HLS PIPELINE II=2
        keyval vi = src[i];
        keyval vj = src[j];
        if (vi.value <= vj.value) {
            ++i;
            dst[k++] = vi;
        } else {
            ++j;
            dst[k++] = vj;
        }
    }
    while(i < mid) {
        #pragma HLS PIPELINE II=1
        dst[k++] = src[i++];
    }
    while(j < right) {
        #pragma HLS PIPELINE II=1
        dst[k++] = src[j++];
    }
}

void merge_sort(
    const keyval* pre_groups,
    const int* accurate_sizes,
    const int num_groups,
    stream<keyval> &stream_out
) {
    #pragma HLS INLINE off

    keyval buf0[N];
    keyval buf1[N];
    #pragma HLS BIND_STORAGE variable=buf0 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=buf1 type=ram_2p impl=bram

    for (int i = 0; i < num_groups; ++i) {
        const int accurate_size = accurate_sizes[i];
        if (accurate_size <= 1) {
            if (accurate_size == 1) stream_out.write(pre_groups[i * N]);
            continue;
        }

        for (int j = 0; j < accurate_size; ++j) {
            #pragma HLS PIPELINE II=1
            buf0[j] = pre_groups[i * N + j];
        }

        bool exchange = true;
        for (int width = 1; width < accurate_size; width <<= 1) {
            for (int left = 0; left < accurate_size; left += 2 * width) {
                int mid = left + width;
                int right = left + 2 * width;
                if (mid > accurate_size) mid = accurate_size;
                if (right > accurate_size) right = accurate_size;

                if (exchange)
                    merge_unit(buf0, left, mid, right, buf1);
                else
                    merge_unit(buf1, left, mid, right, buf0);
            }
            exchange = !exchange;
        }

        if (exchange) {
            for (int j = 0; j < accurate_size; ++j) {
                #pragma HLS PIPELINE II=1
                stream_out.write(buf0[j]);
            }
        } else {
            for (int j = 0; j < accurate_size; ++j) {
                #pragma HLS PIPELINE II=1
                stream_out.write(buf1[j]);
            }
        }
    }
    stream_out.write({-1, 0});
}

void data_input(
    stream<keyval> &kv_in,
    const md_f8_4* md_in,
    stream <md_n1f8> &md_out
) {
    #pragma HLS INLINE off
    while(true) {
        #pragma HLS PIPELINE II=1
        keyval temp = kv_in.read();
        if (temp.key == -1)
            break;
        md_out.write({temp.key, md_in[temp.key]});
    }
    md_out.write({-1, 0});
}

bool in_frustum(
    const data_t3 p,
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    data_t &m_w,
    data_t2 &ndc_xy,
    data_t2 &p_view,
    data_t &depth
) {
    #pragma HLS INLINE 
    #pragma HLS ARRAY_PARTITION variable=viewmatrix type=complete
    #pragma HLS ARRAY_PARTITION variable=projmatrix type=complete

    const data_t x = computeVecter(p, projmatrix, 0);
    const data_t y = computeVecter(p, projmatrix, 1);  
    const data_t w = computeVecter(p, projmatrix, 3);

    data_t view_x = computeVecter(p, viewmatrix, 0);
    data_t view_y = computeVecter(p, viewmatrix, 1);
    depth = computeVecter(p, viewmatrix, 2);

    m_w = hls::recipf(w + FP_MIN);
    ndc_xy = {x * m_w, y * m_w};
    p_view = {view_x, view_y};

    bool valid = (ndc_xy.x < -FP_1o3 || ndc_xy.x > FP_1o3 || ndc_xy.y < -FP_1o3 || ndc_xy.y > FP_1o3);
    return valid;
}

void computeCov2D(
    const data_t depth, 
    const data_t focal_x, 
    const data_t focal_y, 
    const data_t radius, 
    const data_t2 p_view,
    data_t3 &cov2D
) {
    #pragma HLS INLINE
    
    const data_t depth2 = depth * depth;
    const data_t depth4 = depth2 * depth2;
    const data_t radius2 = radius * radius;
    const data_t temp1 = p_view.x * p_view.x + depth2;
    const data_t temp2 = p_view.x * p_view.y;
    const data_t temp3 = p_view.y * p_view.y + depth2;

    const data_t base_variance = radius2 / depth4; 
    const data_t temp4 = focal_x * focal_x;
    const data_t temp5 = focal_x * focal_y;
    const data_t temp6 = focal_y * focal_y;
    const data_t temp7 = base_variance * temp1;
    const data_t temp8 = base_variance * temp2;
    const data_t temp9 = base_variance * temp3;

    cov2D.x = temp4 * temp7 + FP_0o3;
    cov2D.y = temp5 * temp8;
    cov2D.z = temp6 * temp9 + FP_0o3;
}

data_t ndc2Pix(const data_t v, const data_t S)
{
    #pragma HLS INLINE
	return ((v + FP_1) * S - FP_1) * FP_0o5;
}

void getRect(
    const data_t2 p, 
    const data_t max_radius, 
    int4 &rect
) {
    #pragma HLS INLINE

    rect.x_min = min(W-1, max(0, int(p.x - max_radius)));
    rect.y_min = min(H-1, max(0, int(p.y - max_radius)));
    rect.x_max = min(W-1, max(0, int(hls::ceil(p.x + max_radius))));
    rect.y_max = min(H-1, max(0, int(hls::ceil(p.y + max_radius))));
}

bool computeConic (
    const data_t3 cov2D,
    const data_t opacity,
    data_t& det,
    data_t4& conic
) {
    #pragma HLS INLINE

    const data_t temp1 = cov2D.x * cov2D.z;
    const data_t temp2 = cov2D.y * cov2D.y;

    det = temp1 - temp2;
    const data_t det_inv = hls::recipf(det);
    conic.x = cov2D.z * det_inv;
    conic.y = -cov2D.y * det_inv;
    conic.z = cov2D.x * det_inv;
    conic.w = opacity;

    return (det == FP_0);
}

bool computeRadius (
    const data_t3 cov2D,
    const data_t det,
    const data_t opacity,
    data_t& my_radius
) {
    #pragma HLS INLINE

    const data_t mid = FP_0o5 * (cov2D.x + cov2D.z);
    const data_t temp = (mid * mid) - det;
    const data_t lambda = mid + hls::sqrt(hls::max(FP_0o1, temp));
    const data_t ln = hls::logf(255.0f * opacity);
    my_radius = hls::ceil(hls::sqrt(2.0f * ln * lambda));

    return (opacity < FP_1l255);
}

bool computeRect (
    const data_t2 ndc_xy,
    const data_t my_radius,
    data_t2& point_xy,
    int4& rect
) {
    #pragma HLS INLINE

    point_xy.x = ndc2Pix(ndc_xy.x, FP_W);
    point_xy.y = ndc2Pix(ndc_xy.y, FP_H);
    getRect(point_xy, my_radius, rect);

    const int temp1 = rect.x_max - rect.x_min;
    const int temp2 = rect.y_max - rect.y_min;
    const int tiles_touched = temp1 * temp2;

    return (tiles_touched == 0);
}

void preprocess(
    stream<md_n1f8> &md_in,
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    const data_t focal_x, const data_t focal_y,
    stream<afr> &stream_out,
    stream<md_f8_1> &md1_out,
    stream<md_f8_2> &md2_out
) {
    #pragma HLS INLINE off

    preprocess_LOOP: while (true) {
        #pragma HLS PIPELINE II=1
        const md_n1f8 temp = md_in.read();
        if (temp.orig_idx == -1)
            break;

        const int orig_idx = temp.orig_idx;
        const data_t3 mean3D = temp.md_temp.means3D;
        const data_t3 color = temp.md_temp.color;
        const data_t opacity = temp.md_temp.opacity;
        const data_t radius = temp.md_temp.radius;

        data_t w;
        data_t2 ndc_xy;
        data_t2 p_view;
        data_t depth;
        data_t det;
        data_t3 cov2D;
        data_t4 conic;
        data_t my_radius;
        data_t2 point_xy;
        int4 rect;

        if (in_frustum(mean3D, viewmatrix, projmatrix, w, ndc_xy, p_view, depth))
            continue;

        computeCov2D(depth, focal_x, focal_y, radius, p_view, cov2D);

        if (computeConic(cov2D, opacity, det, conic))
            continue;

        if (computeRadius(cov2D, det, opacity, my_radius))
            continue;

        if (computeRect(ndc_xy, my_radius, point_xy, rect))
            continue;
        
        stream_out.write({orig_idx, conic, color, point_xy, depth, rect});
        md1_out.write({{color.x, color.y, color.z, depth}, conic});
        md2_out.write({mean3D, radius, cov2D, w});
    }
    stream_out.write({-1, 0, 0, 0, 0, 0});
}

void init_renderData(
    // bool t_map[H][W],
    ren pix_data[H][W]
) {
    #pragma HLS INLINE off

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            #pragma HLS PIPELINE II=1
            // t_map[y][x] = 0;
            pix_data[y][x] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
}

// int early_termination(
//     const data_t2 point_xy,
//     const bool t_map[H][W]
// ) {
//     #pragma HLS INLINE 

//     int total = 0;
//     int m_base = min(W-3, max(0, int(point_xy.x) - 1));
//     int n_base = min(H-3, max(0, int(point_xy.y) - 1));
//     for (int n = 0; n < 3; ++n) {
//         for (int m = 0; m < 3; ++m) {
//             #pragma HLS UNROLL
//             int m_idx = m_base + m;
//             int n_idx = n_base + n;
//             total += t_map[n_idx][m_idx];
//         }
//     }

//     return total;
// }

void computeRender(
    const int4 rect,
    const data_m2 point_xy,
    const data_m4 coldep,
    const data_m4 conic,
    ren pix_data[H][W]
    // bool t_map[H][W]
) {
    #pragma HLS INLINE
    for (int y_base = rect.y_min; y_base < rect.y_max; y_base += NUM_RENDERERS) {
        for (int x_base = rect.x_min; x_base < rect.x_max; x_base += NUM_RENDERERS) {
            #pragma HLS PIPELINE II=1

            for (int j = 0; j < NUM_RENDERERS; ++j) {
                for (int i = 0; i < NUM_RENDERERS; ++i) {
                    int x = x_base + i;
                    int y = y_base + j;
                    if (x >= rect.x_max || y >= rect.y_max)
                        continue;

                    const int idx = y * W + x;
                    ren temp = pix_data[y][x];
                    data_m t = temp.t;
                    data_m r = temp.r;
                    data_m g = temp.g;
                    data_m b = temp.b;
                    data_m d = temp.d;
                    data_m s = temp.accum_r;

                    data_m2 dis = {point_xy.x - x, point_xy.y - y};
                    data_m temp1 = conic.x * dis.x * dis.x;
                    data_m temp2 = conic.z * dis.y * dis.y;
                    data_m temp3 = conic.y * dis.x * dis.y;
                    data_m temp4 = temp1 + temp2;
                    data_m temp5 = -FP_0o5 * temp4;
                    data_m power = temp5 - temp3;

                    data_m alpha = min(FP_0o99, data_m(conic.w * hls::exp(power)));
                    data_m T = alpha * t;

                    r += coldep.x * T;
                    g += coldep.y * T;
                    b += coldep.z * T;
                    d += coldep.w * T;
                    s += T;
                    t *= (data_m(1.0f) - alpha);

                    pix_data[y][x] = {t, r, g, b, d, s, 0, 0, 0};

                    // if (t < FP_TMIN)
                    //     t_map[y][x] = 1;
                }
            }
        }
    }
}

void render_output(
    const ren pix_data[H][W],
    data_t3* out_colors,
    data_t* out_depths,
    data_t* out_ss,
    data_t* final_Ts
) {
    #pragma HLS INLINE off

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            #pragma HLS PIPELINE II=2
            int idx = y * W + x;
            ren temp = pix_data[y][x];
            data_t t = data_t(temp.t);
            data_t r = data_t(temp.r);
            data_t g = data_t(temp.g);
            data_t b = data_t(temp.b);
            data_t d = data_t(temp.d);
            data_t s = data_t(temp.accum_r);

            out_colors[idx] = {r, g, b};
            out_depths[idx] = d;
            out_ss[idx] = s;
            final_Ts[idx] = t;
        }
    }
}

void render(
    stream<afr> &stream_in,
    ren pix_data[H][W],
    // bool t_map[H][W],
    stream<md_n6f2> &md_out,
    data_t3* out_colors,
    data_t* out_depths,
    data_t* out_ss,
    data_t* final_Ts
) {
    #pragma HLS INLINE off
    #pragma HLS DEPENDENCE variable=pix_data inter false

    int count = 0;
    while (true) {
        const afr temp = stream_in.read();
        int index = temp.index;
        data_m4 conic = {data_m(temp.conic.x), data_m(temp.conic.y), data_m(temp.conic.z), data_m(temp.conic.w)};      
        data_m4 coldep = {data_m(temp.color.x), data_m(temp.color.y), data_m(temp.color.z), data_m(temp.depth)};
        data_m2 point_xy_m = {data_m(temp.point_xy.x), data_m(temp.point_xy.y)};
        data_t2 point_xy = temp.point_xy;
        int4 rect = temp.rect;

        if (index == -1)
            break;

        // int total = early_termination(point_xy, t_map);
        int total = 0;

        if (total > 5) {
            md_out.write({index, 0, rect, point_xy});
        } else {
            md_out.write({index, 1, rect, point_xy});
            computeRender(rect, point_xy_m, coldep, conic, pix_data);
        }
    }
    render_output(pix_data, out_colors, out_depths, out_ss, final_Ts);
    md_out.write({-1, 0, 0, 0});
}

void data_output(
    stream<md_n6f2> &md1_in,
    stream<md_f8_1> &md2_in,
    stream<md_f8_2> &md3_in,
    md_n6f2* md1_out,
    md_f8_1* md2_out,
    md_f8_2* md3_out,
    int &total_count
) {
    #pragma HLS INLINE off
    int count = 0;
    while (true) {
        #pragma HLS PIPELINE II=1
        const md_n6f2 temp_1 = md1_in.read();
        if (temp_1.orig_idx == -1)
            break;
        const md_f8_1 temp_2 = md2_in.read();
        const md_f8_2 temp_3 = md3_in.read();

        md1_out[count] = temp_1;
        md2_out[count] = temp_2;
        md3_out[count] = temp_3;

        ++count;
    }
    total_count = count;
}

void forward_dataflow(
    const keyval* pre_groups,
    const md_f8_4* md_1,
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    const data_t focal_x, const data_t focal_y,
    const int* accurate_sizes,
    const int num_groups,
    ren pix_data[H][W],
    // bool t_map[H][W],
    int &total_count,
    md_n6f2* md_2,
    md_f8_1* md_3,
    md_f8_2* md_4,
    data_t3* out_colors,
    data_t* out_depths,
    data_t* out_ss,
    data_t* final_Ts
) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    stream<keyval, 64> stream_kv;
    stream<afr, 64> stream_afr;
    stream<md_n1f8, 64> stream_md1;
    stream<md_f8_1, 64> stream_md2;
    stream<md_f8_2, 64> stream_md3;
    stream<md_n6f2, 64> stream_md4;
    #pragma HLS BIND_STORAGE variable=stream_kv type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_afr type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md1 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md2 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md3 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md4 type=fifo impl=srl

    merge_sort(
        pre_groups,
        accurate_sizes,
        num_groups,
        stream_kv
    );

    data_input(
        stream_kv,
        md_1,
        stream_md1
    );

    preprocess(
        stream_md1,
        viewmatrix,
        projmatrix,
        focal_x, focal_y,
        stream_afr,
        stream_md2,
        stream_md3
    );

    render(
        stream_afr,
        pix_data,
        // t_map,
        stream_md4,
        out_colors,
        out_depths,
        out_ss,
        final_Ts
    );

    data_output(
        stream_md4,
        stream_md2,
        stream_md3,
        md_2,
        md_3,
        md_4,
        total_count
    );
}

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
) {
    #pragma HLS INLINE off

    // bool t_map[H][W];
    // #pragma HLS BIND_STORAGE variable=t_map type=ram_1p impl=bram
    // #pragma HLS ARRAY_PARTITION variable=t_map dim=1 cyclic factor=3
    // #pragma HLS ARRAY_PARTITION variable=t_map dim=2 cyclic factor=3

    init_renderData(pix_data);

    forward_dataflow(
        pre_groups,
        md_1,
        viewmatrix,
        projmatrix,
        focal_x, focal_y,
        accurate_sizes,
        num_groups,
        pix_data,
        // t_map,
        total_count,
        md_2,
        md_3,
        md_4,
        out_colors,
        out_depths,
        out_ss,
        final_Ts
    );
}