#include "backward_config.h"
using namespace hls;

data_t computeVecter3(
    const data_t x,
    const data_t y,
    const data_t z,
    const data_t m[16],
    const int idx
) {
    #pragma HLS INLINE

    const data_t temp1 = m[idx] * x; 
    const data_t temp2 = m[1 + idx] * y + temp1; 
    const data_t sum = m[2 + idx] * z + temp2;
    
    return sum;
}

data_t compute_dL_dmean(
    const data_t m[16],
    const data_t m_w,
    const data_t mul1,
    const data_t mul2,
    const data_t2 dL_dmean2D,
    const int idx
) {
    #pragma HLS INLINE

    const data_t temp1 = -m[idx + 3] * mul1;
    const data_t temp2 = m[idx] * m_w + temp1;
    const data_t temp3 = temp2 * dL_dmean2D.x;
    const data_t temp4 = -m[idx + 3] * mul2;
    const data_t temp5 = m[idx + 1] * m_w + temp4;
    const data_t temp6 = temp5 * dL_dmean2D.y;
    const data_t sum = temp3 + temp6;

    return sum;
}

void init_renderData(
    const bool4* dL_dcoldeps,
    const data_t* final_Ts,
    const bool* depth_mask,
    const bool* color_mask,
    ren pix_data[H][W],
    m1b2 last_alphas[H][W],
    bool4 dL_dcoldep[H][W]
) {
    #pragma HLS INLINE off
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            #pragma HLS PIPELINE II=1
            int idx = y * W + x;
            last_alphas[y][x] = {0, depth_mask[idx], color_mask[idx]};
            dL_dcoldep[y][x] = dL_dcoldeps[idx];
            pix_data[y][x] = {data_m(final_Ts[idx]), 0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
}

void data_input(
    const md_n6f2* md_1, 
    const md_f8_1* md_2,
    const md_f8_2* md_3,
    const int size,
    stream<md_n2> &md1_out,
    stream<md_n4f10> &md2_out,
    stream<md_f8_2> &md3_out
) {
    #pragma HLS INLINE off
    for (int idx = size - 1; idx >= 0; --idx) {
        #pragma HLS PIPELINE II=1
        md_n6f2 temp_1 = md_1[idx];
        md_f8_1 temp_2 = md_2[idx];
        md_f8_2 temp_3 = md_3[idx];
        md1_out.write({temp_1.orig_idx, temp_1.n_contrib});
        if (temp_1.n_contrib == 1) {
            md2_out.write({temp_1.rect, temp_1.point_xy, temp_2.coldep, temp_2.conic});  
            md3_out.write(temp_3);        
        }
    }
    const int4 end = {-1, 0, 0, 0};
    md2_out.write({end, 0, 0, 0});
}

void compute_num_add_out(
    const int4 rect,
    stream<int> &num_add_out
) {
    #pragma HLS INLINE
    int temp_x = rect.x_max - rect.x_min + 1;
    int temp_y = rect.y_max - rect.y_min + 1;
    int num_x = temp_x / NUM_RENDERERS;
    int num_y = temp_y / NUM_RENDERERS;
    int num = num_x * num_y;
    if (num != 0) 
        num_add_out.write(num);
}

void compute_renderGrad(
    ren pix_data[H][W],
    const int4 rect,
    const bool4 dL_dcoldeps[H][W],
    m1b2 last_alphas[H][W],
    const data_m2 point_xy,
    const data_m4 conic,
    const data_m4 coldep,
    stream<md_f10> &grads_out
) {
    #pragma HLS INLINE
    for (int y_base = rect.y_min; y_base < rect.y_max; y_base+=NUM_RENDERERS) {
        for (int x_base = rect.x_min; x_base < rect.x_max; x_base+=NUM_RENDERERS) {
            #pragma HLS PIPELINE II=1

            data_m3 dL_dcolor = {0, 0, 0};
            data_m dL_ddepth = 0;
            data_m2 dL_dmean2D = {0, 0};
            data_m3 dL_dconic2D = {0, 0, 0};
            data_m dL_dopacity = 0;

            for (int j = 0; j < NUM_RENDERERS; ++j) {
                for (int i = 0; i < NUM_RENDERERS; ++i) {
                    int x = x_base + i;
                    int y = y_base + j;
                    if (x >= rect.x_max || y >= rect.y_max)
                        continue;

                    const ren temp = pix_data[y][x];
                    data_m t = temp.t;
                    data_m last_r = temp.r;
                    data_m last_g = temp.g;
                    data_m last_b = temp.b;
                    data_m last_d = temp.d;
                    data_m accum_r = temp.accum_r;
                    data_m accum_g = temp.accum_g;
                    data_m accum_b = temp.accum_b;
                    data_m accum_d = temp.accum_d;

                    const bool4 temp_2 = dL_dcoldeps[y][x];
                    data_m4 dL_dcoldep = {
                        temp_2.r ? data_m(color_mul) : data_m(-color_mul),
                        temp_2.g ? data_m(color_mul) : data_m(-color_mul),
                        temp_2.b ? data_m(color_mul) : data_m(-color_mul),
                        temp_2.d ? data_m(depth_mul) : data_m(-depth_mul),
                    };

                    m1b2 temp_3 = last_alphas[y][x];
                    data_m last_alpha = temp_3.last_alpha;
                    if (temp_3.is_color_zero)
                        dL_dcoldep.x = dL_dcoldep.y = dL_dcoldep.z = HFP_0;
                    if (temp_3.is_depth_zero)
                        dL_dcoldep.w = HFP_0;

                    const data_m2 dis = {point_xy.x - x, point_xy.y - y};
                    const data_m temp1 = conic.x * dis.x * dis.x;
                    const data_m temp2 = conic.z * dis.y * dis.y;
                    const data_m temp3 = conic.y * dis.x * dis.y;
                    const data_m temp4 = -HFP_0o5 * (temp1 + temp2);

                    const data_m power = temp4 - temp3;
                    const data_m G = hls::exp(power);
                    data_m alpha = min(HFP_0o99, data_m(conic.w * G));

                    t /= (HFP_1 - alpha);
                    const data_m dchannel_dcolor = alpha * t;

                    dL_dcolor.x += dchannel_dcolor * dL_dcoldep.x;
                    dL_dcolor.y += dchannel_dcolor * dL_dcoldep.y;
                    dL_dcolor.z += dchannel_dcolor * dL_dcoldep.z;
                    dL_ddepth   += dchannel_dcolor * dL_dcoldep.w;

                    const data_m temp5 = data_m(1.0f) - last_alpha;
                    const data_m temp6 = temp5 * accum_r;
                    const data_m temp7 = temp5 * accum_g;
                    const data_m temp8 = temp5 * accum_b;
                    const data_m temp9 = temp5 * accum_d;
                    accum_r = last_alpha * last_r + temp6;
                    accum_g = last_alpha * last_g + temp7;
                    accum_b = last_alpha * last_b + temp8;
                    accum_d = last_alpha * last_d + temp9;
                    last_r = coldep.x * HFP_256;
                    last_g = coldep.y * HFP_256;
                    last_b = coldep.z * HFP_256;
                    last_d = coldep.w * HFP_256;

                    last_alphas[y][x] = {alpha, temp_3.is_depth_zero, temp_3.is_color_zero};
                    pix_data[y][x] = {t, last_r, last_g, last_b, last_d, accum_r, accum_g, accum_b, accum_d};

                    const data_m temp10 = (last_r - accum_r) * dL_dcoldep.x;
                    const data_m temp11 = (last_g - accum_g) * dL_dcoldep.y;
                    const data_m temp12 = (last_b - accum_b) * dL_dcoldep.z;
                    const data_m temp13 = (last_d - accum_d) * dL_dcoldep.w;
                    const data_m temp14 = temp10 + temp11;
                    const data_m temp15 = temp12 + temp13;
                    data_m dL_dalpha;
                    dL_dalpha = (temp14 + temp15) * t;

                    const data_m dL_dG = conic.w * dL_dalpha;
                    const data_m gdx = G * dis.x;
                    const data_m gdy = G * dis.y;
                    const data_m temp16 = -gdx * conic.x;
                    const data_m temp17 = -gdy * conic.y;
                    const data_m temp18 = -gdy * conic.z;
                    const data_m temp19 = -gdx * conic.y;
                    const data_m dG_ddelx = temp16 + temp17;
                    const data_m dG_ddely = temp18 + temp19;

                    const data_m temp20 = dL_dG * dG_ddelx * DDELX_DX;
                    const data_m temp21 = dL_dG * dG_ddely * DDELY_DY;
                    dL_dmean2D.x += temp20;
                    dL_dmean2D.y += temp21;

                    const data_m temp22 = -HFP_0o5 * gdx;
                    const data_m temp23 = -HFP_0o5 * gdy;
                    const data_m temp24 = dis.x * dL_dG;
                    const data_m temp25 = dis.y * dL_dG;
                    dL_dconic2D.x += temp22 * temp24;
                    dL_dconic2D.y += temp22 * temp25;
                    dL_dconic2D.z += temp23 * temp25;

                    dL_dopacity += G * dL_dalpha;
                }
            }            
            grads_out.write({dL_dcolor, dL_ddepth, dL_dmean2D, dL_dconic2D, dL_dopacity});
        }
    }
}

void render_backward(
    stream<md_n4f10> &md_in,
    ren pix_data[H][W], 
    bool4 dL_dcoldeps[H][W],
    m1b2 last_alphas[H][W],
    stream<int> &num_add_out,
    stream<md_f10> &grads_out
) {
    #pragma HLS INLINE off
    #pragma HLS DEPENDENCE variable=pix_data    inter false
    #pragma HLS DEPENDENCE variable=last_alphas inter false

    while (true) {
        const md_n4f10 temp = md_in.read(); 
        int4 rect = temp.rect; 
        data_m2 point_xy = {data_m(temp.point_xy.x), data_m(temp.point_xy.y)}; 
        data_m4 conic = {data_m(temp.conic.x), data_m(temp.conic.y), data_m(temp.conic.z), data_m(temp.conic.w)}; 
        data_m4 coldep = { 
            data_m(temp.coldep.x), 
            data_m(temp.coldep.y), 
            data_m(temp.coldep.z), 
            data_m(temp.coldep.w) 
        }; 

        if (rect.x_min == -1)
            break;

        compute_num_add_out(rect, num_add_out);

        compute_renderGrad(pix_data, rect, dL_dcoldeps, last_alphas, point_xy, conic, coldep, grads_out);
    }
    num_add_out.write(-1);
}

data_m computeAdd(
    const data_m x,
    const data_m y,
    const data_m z,
    const data_m w
) {
    #pragma HLS INLINE

    const data_m temp1 = x + y;
    const data_m temp2 = z + w;

    return temp1 + temp2;
}

void compute_total_grads(
    const int num,
    stream<md_f10> &grads_in,
    data_m3 &dL_dcolor_total,
    data_m &dL_ddepth_total,
    data_m2 &dL_dmean2D_total,
    data_m3 &dL_dconic2D_total,
    data_m &dL_dopacity_total
) {
    #pragma HLS INLINE
    for (int i = 0; i < num; i+=4) {
        #pragma HLS PIPELINE II=4

        data_m3 dL_dcolor[4] = {0};
        data_m dL_ddepth[4] = {0};
        data_m2 dL_dmean2D[4] = {0};
        data_m3 dL_dconic2D[4] = {0};
        data_m dL_dopacity[4] = {0};

        for (int j = 0; j < 4; ++j) {
            int idx = i + j;
            if (idx >= num)
                continue;
            md_f10 temp = grads_in.read();
            dL_dcolor[j] = temp.dL_dcolor;
            dL_ddepth[j] = temp.dL_ddepth;
            dL_dmean2D[j] = temp.dL_dmean2D;
            dL_dconic2D[j] = temp.dL_dconic2D;
            dL_dopacity[j] = temp.dL_dopacity;
        }

        dL_dcolor_total.x += computeAdd(dL_dcolor[0].x, dL_dcolor[1].x, dL_dcolor[2].x, dL_dcolor[3].x);
        dL_dcolor_total.y += computeAdd(dL_dcolor[0].y, dL_dcolor[1].y, dL_dcolor[2].y, dL_dcolor[3].y);
        dL_dcolor_total.z += computeAdd(dL_dcolor[0].z, dL_dcolor[1].z, dL_dcolor[2].z, dL_dcolor[3].z);
        dL_ddepth_total += computeAdd(dL_ddepth[0], dL_ddepth[1], dL_ddepth[2], dL_ddepth[3]);
        dL_dmean2D_total.x += computeAdd(dL_dmean2D[0].x, dL_dmean2D[1].x, dL_dmean2D[2].x, dL_dmean2D[3].x);
        dL_dmean2D_total.y += computeAdd(dL_dmean2D[0].y, dL_dmean2D[1].y, dL_dmean2D[2].y, dL_dmean2D[3].y);
        dL_dconic2D_total.x += computeAdd(dL_dconic2D[0].x, dL_dconic2D[1].x, dL_dconic2D[2].x, dL_dconic2D[3].x);
        dL_dconic2D_total.y += computeAdd(dL_dconic2D[0].y, dL_dconic2D[1].y, dL_dconic2D[2].y, dL_dconic2D[3].y);
        dL_dconic2D_total.z += computeAdd(dL_dconic2D[0].z, dL_dconic2D[1].z, dL_dconic2D[2].z, dL_dconic2D[3].z);
        dL_dopacity_total += computeAdd(dL_dopacity[0], dL_dopacity[1], dL_dopacity[2], dL_dopacity[3]);
    }
}

void grads_adder(
    stream<int> &num_in,
    stream<md_f10> &grads_in,
    stream<md_f4_1> &md1_out,
    stream<md_f6> &md2_out
) {
    #pragma HLS INLINE off

    while (true) {
        int num = num_in.read();
        if (num == -1)
            break;

        data_m3 dL_dcolor_total = {0, 0, 0};
        data_m dL_ddepth_total = 0;
        data_m2 dL_dmean2D_total = {0, 0};
        data_m3 dL_dconic2D_total = {0, 0, 0};
        data_m dL_dopacity_total = 0;

        compute_total_grads(num, grads_in, dL_dcolor_total, dL_ddepth_total, dL_dmean2D_total, dL_dconic2D_total, dL_dopacity_total);

        md1_out.write({{data_t(dL_dcolor_total.x) * d_div, data_t(dL_dcolor_total.y) * d_div, data_t(dL_dcolor_total.z) * d_div}, data_t(dL_dopacity_total) * d_div_plus});
        md2_out.write({data_t(dL_ddepth_total) * d_div, {data_t(dL_dmean2D_total.x) * d_div_plus, data_t(dL_dmean2D_total.y) * d_div_plus}, {data_t(dL_dconic2D_total.x) * d_div_plus, data_t(dL_dconic2D_total.y) * d_div_plus, data_t(dL_dconic2D_total.z) * d_div_plus}});
    }
    const data_t temp = 233.0f;
    md2_out.write({temp, 0, 0});
}

void preprocess_backward(
    stream<md_f6> &md1_in,
    stream<md_f8_2> &md2_in,
    const data_t focal_x, data_t focal_y,
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    stream<md_f4_2> &md_out
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=viewmatrix type=complete
    #pragma HLS ARRAY_PARTITION variable=projmatrix type=complete

    while (true) {
        #pragma HLS PIPELINE II=2

        const md_f6 temp_1 = md1_in.read();
        const data_t3 dL_dconic = temp_1.dL_dconic2D;
        const data_t dL_ddepth = temp_1.dL_ddepth;
        const data_t2 dL_dmean2D = temp_1.dL_dmean2D;

        stream<data_t,1> test;
        #pragma HLS BIND_STORAGE variable=test type=fifo impl=srl
        test.write(dL_ddepth);
        if (test.read() == 233.0f)
            break;

        const md_f8_2 temp_2 = md2_in.read();
        const data_t a = temp_2.cov2D.x, b = temp_2.cov2D.y, c = temp_2.cov2D.z;
        const data_t3 p = temp_2.means3D;
        const data_t radius = temp_2.radius;
        const data_t m_w = temp_2.w;
        const data_t m_w2 = m_w * m_w;

        data_t3 t;
        t.x = computeVecter(p, viewmatrix, 0);
        t.y = computeVecter(p, viewmatrix, 1);
        t.z = computeVecter(p, viewmatrix, 2);

        const data_t temp1 = - b * b;
        const data_t denom = a * c + temp1;
        const data_t temp2 = (denom * denom) + 0.0000001f;
        const data_t denom2inv = hls::recipf(temp2);
        data_t dL_da = 0, dL_db = 0, dL_dc = 0;

        const data_t focal_xy = focal_x * focal_y;
        const data_t focal_xx = focal_x * focal_x;
        const data_t focal_yy = focal_y * focal_y;
        const data_t tz2 = t.z * t.z;
        const data_t tz4 = tz2 * tz2;
        const data_t tz4inv = hls::recipf(tz4);
        data_t dL_dvariance = 0, dL_dradius = 0;

        if (denom2inv != 0) {
            const data_t term1 = -c * c * dL_dconic.x;
            const data_t term2 = -a * a * dL_dconic.z;
            const data_t term3 = b * c;
            const data_t term4 = a * b;
            const data_t term5 = denom - a * c;
            const data_t term6 = 2 * dL_dconic.y * term3;
            const data_t term7 = 2 * dL_dconic.y * term4;
            const data_t term8 = dL_dconic.z * term5;
            const data_t term9 = dL_dconic.x * term5;
            const data_t term10 = term1 + term6 + term8;
            const data_t term11 = term2 + term7 + term9;
            const data_t term12 = term3 * dL_dconic.x;
            const data_t term13 = 2 * b * b;
            const data_t term14 = (denom + term13) * dL_dconic.y;
            const data_t term15 = term4 * dL_dconic.z;
            const data_t term16 = term12 - term14 + term15;
            dL_da = denom2inv * term10;
            dL_dc = denom2inv * term11;
            dL_db = denom2inv * 2 * term16;

            const data_t term17 = tz2 + t.x * t.x;
            const data_t term18 = focal_xx * term17 * dL_da;
            const data_t term19 = tz2 + t.y * t.y;
            const data_t term20 = focal_yy * term19 * dL_dc;
            const data_t term21 = focal_xy * t.x;
            const data_t term22 = t.y * dL_db;
            const data_t term23 = term21 * term22;
            const data_t term24 = 2 * radius;
            dL_dvariance = term18 + term20 + term23; 
            dL_dradius = term24 * tz4inv * dL_dvariance;
        } else {
            dL_dradius = 0;
        }

        const data_t variance = (radius * radius) * tz4inv; 
        const data_t tzinv = hls::recipf(t.z);
        data_t dL_dtx = 0, dL_dty = 0, dL_dtz = 0;
        
        const data_t temp3 = focal_xx * t.x * dL_da;
        const data_t temp4 = focal_xy * t.y * dL_db;
        const data_t temp5 = 2 * temp3 + temp4;
        const data_t temp6 = focal_yy * t.y * dL_dc;
        const data_t temp7 = focal_xy * t.x * dL_db;
        const data_t temp8 = 2 * temp6 + temp7;
        const data_t temp9 = -4 * focal_xy * t.x;
        const data_t temp10 = t.y * tzinv * dL_db;
        const data_t temp11 = temp9 * temp10;
        const data_t temp12 = t.x * t.x;
        const data_t temp13 = 4 * temp12 * tzinv;
        const data_t temp14 = 2 * t.z + temp13;
        const data_t temp15 = - focal_xx * temp14 * dL_da;
        const data_t temp16 = t.y * t.y;
        const data_t temp17 = 4 * temp16 * tzinv;
        const data_t temp18 = 2 * t.z + temp17;
        const data_t temp19 = - focal_yy * temp18 * dL_dc;
        const data_t temp20 = temp11 + temp15 + temp19;
        dL_dtx = variance * temp5;
        dL_dty = variance * temp8;
        dL_dtz = variance * temp20;

        data_t dL_dpx = 0, dL_dpy = 0, dL_dpz = 0;
        dL_dpx = computeVecter3(dL_dtx, dL_dty, dL_dtz, viewmatrix, 0);
        dL_dpy = computeVecter3(dL_dtx, dL_dty, dL_dtz, viewmatrix, 4);
        dL_dpz = computeVecter3(dL_dtx, dL_dty, dL_dtz, viewmatrix, 8);

        data_t3 dL_dmean3D = {dL_dpx, dL_dpy, dL_dpz};
        data_t3 dL_dmean;

        data_t mul1 = computeVecter(p, projmatrix, 0) * m_w2;
        data_t mul2 = computeVecter(p, projmatrix, 1) * m_w2;
        dL_dmean.x = compute_dL_dmean(projmatrix, m_w, mul1, mul2, dL_dmean2D, 0);
        dL_dmean.y = compute_dL_dmean(projmatrix, m_w, mul1, mul2, dL_dmean2D, 4);
        dL_dmean.z = compute_dL_dmean(projmatrix, m_w, mul1, mul2, dL_dmean2D, 8);

        const data_t temp21 = dL_ddepth * viewmatrix[2] + dL_dmean.x;
        const data_t temp22 = dL_ddepth * viewmatrix[6] + dL_dmean.y;
        const data_t temp23 = dL_ddepth * viewmatrix[10] + dL_dmean.z;
        dL_dmean3D.x += temp21;
        dL_dmean3D.y += temp22;
        dL_dmean3D.z += temp23;

        md_out.write({dL_dmean3D, dL_dradius});
    }
}

void data_output(
    stream<md_n2> &md1_in,
    stream<md_f4_1> &md2_in,
    stream<md_f4_2> &md3_in,
    const int size,
    md_f8_3* md_out
) {
    #pragma HLS INLINE off
    back_dataout_LOOP: for (int i = size - 1; i >= 0; --i) {
        #pragma HLS PIPELINE II=1
        const md_n2 temp_1 = md1_in.read();
        if (temp_1.n_contrib == 1) {
            const md_f4_1 temp_2 = md2_in.read();
            const md_f4_2 temp_3 = md3_in.read();
            md_out[temp_1.orig_idx] = {temp_3.dL_dmean3D, temp_3.dL_dradius, temp_2.dL_dcolor, temp_2.dL_dopacity};
        }
    }
}

void backward_dataflow(
    const md_n6f2* md_1, 
    const md_f8_1* md_2,
    const md_f8_2* md_3,
    const data_t focal_x, data_t focal_y,
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    const int size,
    ren pix_data[H][W],
    bool4 dL_dcoldeps[H][W],
    m1b2 last_alphas[H][W],
    md_f8_3* md_4
) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    stream<md_n2, 192> stream_md1;
    stream<md_n4f10, 64> stream_md2;
    stream<md_f8_2, 64> stream_md3;
    stream<md_f10, 64> stream_md4;
    stream<md_f4_1, 192> stream_md5;
    stream<md_f6, 64> stream_md6;
    stream<md_f4_2, 64> stream_md7;
    #pragma HLS BIND_STORAGE variable=stream_md1 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md2 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md3 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md4 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md5 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md6 type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=stream_md7 type=fifo impl=srl
    stream<int, 16> stream_num;

    data_input(
        md_1, 
        md_2,
        md_3,
        size,
        stream_md1,
        stream_md2,
        stream_md3
    );

    render_backward(
        stream_md2,
        pix_data,
        dL_dcoldeps, 
        last_alphas,
        stream_num,
        stream_md4
    ); 

    grads_adder(
        stream_num,
        stream_md4,
        stream_md5,
        stream_md6
    );

    preprocess_backward(
        stream_md6,
        stream_md3,
        focal_x, focal_y,
        viewmatrix,
        projmatrix,
        stream_md7
    );

    data_output(
        stream_md1,
        stream_md5,
        stream_md7,
        size,
        md_4
    );    
}

void backward(
    const md_n6f2* md_1, 
    const md_f8_1* md_2,
    const md_f8_2* md_3,
    const bool4* dL_dcoldeps_in,
    const data_t* final_Ts,
    const bool* depth_mask,
    const bool* color_mask,
    const data_t focal_x, data_t focal_y,
    const data_t viewmatrix[16],
    const data_t projmatrix[16],
    const int size,
    ren pix_data[H][W],
    md_f8_3* md_4
) {
    #pragma HLS INLINE off

    m1b2 last_alphas[H][W];
    #pragma HLS AGGREGATE variable=last_alphas compact=bit
    #pragma HLS BIND_STORAGE variable=last_alphas type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=last_alphas dim=1 cyclic factor=NUM_RENDERERS
    #pragma HLS ARRAY_PARTITION variable=last_alphas dim=2 cyclic factor=NUM_RENDERERS

    bool4 dL_dcoldeps[H][W];
    #pragma HLS AGGREGATE variable=dL_dcoldeps compact=bit
    #pragma HLS BIND_STORAGE variable=dL_dcoldeps type=ram_1p impl=bram
    #pragma HLS ARRAY_PARTITION variable=dL_dcoldeps dim=1 cyclic factor=NUM_RENDERERS
    #pragma HLS ARRAY_PARTITION variable=dL_dcoldeps dim=2 cyclic factor=NUM_RENDERERS

    init_renderData(
        dL_dcoldeps_in,
        final_Ts,
        depth_mask,
        color_mask,
        pix_data,
        last_alphas,
        dL_dcoldeps
    ); 

    backward_dataflow(
        md_1, 
        md_2,
        md_3,
        focal_x, focal_y,
        viewmatrix,
        projmatrix,
        size,
        pix_data,
        dL_dcoldeps,
        last_alphas,
        md_4
    );
}