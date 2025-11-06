import numpy as np
import pynq
# noinspection PyUnresolvedReferences
from pynq import allocate
import time

import struct

class GlobalAddr:
    temp_mem_buffer = allocate((256 * ( 256 * 256 ), 2), dtype=np.float32)
    sortouts_buffer = allocate((256 * ( 256 * 256 ), 2), dtype=np.float32)



def write_64bit_address(ip, offset_low, offset_high, address):
    addr_int = int(address)
    low_32 = addr_int & 0xFFFFFFFF
    high_32 = (addr_int >> 32) & 0xFFFFFFFF

    ip.write(offset_low, low_32)
    ip.write(offset_high, high_32)

def pre_grouping(md_1_for, pre_grouping_ip, raster_settings):
    print("trying to : pre grouping...")
    
    pre_grouping_ip.register_map.CTRL.SW_RESET = 1
    
    # input means3D 
    ta = time.time()
    means_3d = md_1_for[:, :3].clone().contiguous().detach().numpy()#.astype(np.float32)  # (N,3) 
    N = means_3d.shape[0]
    tb = time.time()
    
    means_3d_buffer = allocate((N, 3), dtype=np.float32)
    np.copyto(means_3d_buffer, means_3d)
    # print("means3D",means_3d_buffer,means_3d_buffer.shape)
    write_64bit_address(pre_grouping_ip, 0x18, 0x1c, means_3d_buffer.physical_address) # checked

    # input thresholds
    thresholds_buffer = allocate((256,), dtype=np.float32)
    thresholds_buffer[:] = raster_settings.thresholds
    # print("thresholds",thresholds_buffer)
    write_64bit_address(pre_grouping_ip, 0x24, 0x28, thresholds_buffer.physical_address) # checked
    

    # input size
    pre_grouping_ip.write(0x244, N) # checked
    # is forward ?
    pre_grouping_ip.write(0x24c, 0) # checked
    # is pre_grouping ?
    pre_grouping_ip.write(0x254, 1) # checked
    
    # input viewmatrix
    viewmatrix = raster_settings.viewmatrix.detach().numpy().reshape(-1) # checked
    for i in range(16):
        offset = 0x12c + i * 8
        float_bytes = struct.pack('f', viewmatrix[i])
        int_value = struct.unpack('I', float_bytes)[0]
        pre_grouping_ip.write(offset, int_value)
        
    tc = time.time()

    # init output buffers
    # temp_mem and sortouts
    # temp_mem_buffer = allocate((256 * ( 256 * 256 ), 2), dtype=np.float32)  # 64 * ( 64 * 256 )
    # sortouts_buffer = allocate((256 * ( 256 * 256 ), 2), dtype=np.float32)  # 64 * ( 64 * 256 )
    write_64bit_address(pre_grouping_ip, 0x30, 0x34, GlobalAddr.temp_mem_buffer.physical_address) # checked
    write_64bit_address(pre_grouping_ip, 0x3c, 0x40, GlobalAddr.sortouts_buffer.physical_address) # checked
    # accurate_size_pre
    accurate_size_pre_buffer = allocate((256 * 256 * 1, ), dtype=np.int32) # 64 * 64 * 1
    write_64bit_address(pre_grouping_ip, 0x48, 0x4c, accurate_size_pre_buffer.physical_address) # checked
    
    td = time.time()

    # 缓存同步
    GlobalAddr.sortouts_buffer.flush()
    means_3d_buffer.flush()
    thresholds_buffer.flush()
    
    # 启动
    t1 = time.time()
    pre_grouping_ip.register_map.CTRL.AP_START = 1
    while True:
        if pre_grouping_ip.register_map.CTRL.AP_DONE == 1:
            break
    t2 = time.time()
    
    # 同步输出缓存
    accurate_size_pre_buffer.invalidate()
    GlobalAddr.sortouts_buffer.invalidate()

    # ip done
    print("buffers take :", (t1-ta)*1000, "ms    means3D", (tb-ta)*1000, "ms    allocate", (tc-tb)*1000, "ms    temp_mem_buffer/sortouts_buffer",           (td-tc)*1000, "ms")
    print("Done! pre_grouping_ip takes :", (t2-t1)*1000, "ms")
    num_groups = pre_grouping_ip.register_map.ap_return.ap_return
    #print("accurate sizes :",accurate_size_pre_buffer[:100])
    #print("num_groups :",num_groups)
    return accurate_size_pre_buffer, num_groups

def ip_renderer(
       md_1_for,
       accurate_sizes,   # pynq buffer
       means2D,
       num_groups,
       raster_settings,
       render_ip
):
    ta = time.time()
    print("trying to : forward...")
    render_ip.register_map.CTRL.SW_RESET = 1
    M = 480 * 360
    Size = sum(accurate_sizes)
    ts = time.time()
    # intput pre_groups
    write_64bit_address(render_ip, 0x54, 0x58, GlobalAddr.sortouts_buffer.physical_address) # checked
    
    # intput md_1_for
    md_1_for_arr = md_1_for.detach().numpy()
    md_1_for_buffer = allocate(md_1_for_arr.shape, dtype=np.float32)
    np.copyto(md_1_for_buffer, md_1_for_arr)
    write_64bit_address(render_ip, 0x60, 0x64, md_1_for_buffer.physical_address) # checked

    # input accurate_sizes
    write_64bit_address(render_ip, 0x6c, 0x70, accurate_sizes.physical_address)

    # input viewmatrix & projmatrix
    viewmatrix = raster_settings.viewmatrix.detach().numpy().reshape(-1)
    for i in range(16):  
        offset = 0x12c + i * 8 # checked
        float_bytes = struct.pack('f', viewmatrix[i])
        int_value = struct.unpack('I', float_bytes)[0]
        render_ip.write(offset, int_value)
    # input projmatrix
    projmatrix = raster_settings.projmatrix.detach().numpy().reshape(-1)
    for i in range(16):  
        offset = 0x1ac + i * 8 # checked
        float_bytes = struct.pack('f', projmatrix[i])
        int_value = struct.unpack('I', float_bytes)[0]
        render_ip.write(offset, int_value)

    # input focal x  &  focal y
    focal_x = raster_settings.focal_x
    float_bytes = struct.pack('f', focal_x)
    int_value = struct.unpack('I', float_bytes)[0]
    render_ip.write(0x22c, int_value) # focal_x checked
    focal_y = raster_settings.focal_y
    float_bytes = struct.pack('f', focal_y)
    int_value = struct.unpack('I', float_bytes)[0]
    render_ip.write(0x234, int_value) # focal_y checked

    # input num_group
    render_ip.write(0x23c, int(num_groups)) # checked
    # input size
    render_ip.write(0x244, int(Size)) # checked
    # is forward ?
    render_ip.write(0x24c, 1) # checked
    # is pre_grouping ?
    render_ip.write(0x254, 0) # checked
   
    # init output buffers
    # md_2_for
    md_2_for_buffer = allocate((Size, 8), dtype=np.float32)
    write_64bit_address(render_ip, 0x78, 0x7c, md_2_for_buffer.physical_address)
    # md_3_for
    md_3_for_buffer = allocate((Size, 8), dtype=np.float32)
    write_64bit_address(render_ip, 0x84, 0x88, md_3_for_buffer.physical_address)
    # md_4_for
    md_4_for_buffer = allocate((Size, 8), dtype=np.float32)
    write_64bit_address(render_ip, 0x90, 0x94, md_4_for_buffer.physical_address)
    # out_colors
    out_colors_buffer = allocate((M, 3), dtype=np.float32)
    write_64bit_address(render_ip, 0x9c, 0xa0, out_colors_buffer.physical_address)
    # out_depths
    out_depths_buffer = allocate((M, ), dtype=np.float32)
    write_64bit_address(render_ip, 0xa8, 0xac, out_depths_buffer.physical_address)
    # out_ss
    out_ss_buffer = allocate((M, ), dtype=np.float32)
    write_64bit_address(render_ip, 0xb4, 0xb8, out_ss_buffer.physical_address)
    # final_Ts_for
    final_Ts_for_buffer = allocate((M, ), dtype=np.float32)
    write_64bit_address(render_ip, 0xc0, 0xc4, final_Ts_for_buffer.physical_address)

    md_1_for_buffer.flush()
    accurate_sizes.flush()
    GlobalAddr.sortouts_buffer.flush()
    render_ip.write(0x00, 0x1)

    t1 = time.time()
    while True:
        if render_ip.register_map.CTRL.AP_DONE == 1:
            break
    t2 = time.time()

    md_2_for_buffer.invalidate()
    md_3_for_buffer.invalidate()
    md_4_for_buffer.invalidate()
    out_colors_buffer.invalidate()
    out_depths_buffer.invalidate()
    out_ss_buffer.invalidate()
    final_Ts_for_buffer.invalidate()
    
    print("buffers take :", (t1-ta)*1000, "ms    sum acc_size",(ts-ta)*1000, "ms")
    print("render ip working time : ", (t2 - t1) * 1000, "ms\n")
    total_count = render_ip.register_map.ap_return.ap_return

    return md_2_for_buffer, md_3_for_buffer, md_4_for_buffer, final_Ts_for_buffer, out_colors_buffer, out_depths_buffer, out_ss_buffer, total_count

def ol_forward(
    md_1_for,
    means2D,
    ip,
    raster_settings,
):
    ta = time.time()
    accurate_sizes, num_groups = pre_grouping(
        md_1_for,
        ip,
        raster_settings
    )
    tb = time.time()
    # render_ip = ol.rasterization_1
    md_2_for, md_3_for, md_4_for, final_Ts_for, color, depth, sil, total_count  = ip_renderer(
        md_1_for,
        accurate_sizes,
        means2D,
        num_groups,
        raster_settings,
        ip
    )
    tc = time.time()
    # np.save('im', color)
    
    print("time pg ",1000*(tb-ta),"ms")
    print("time fw ",1000*(tc-tb),"ms")

    return md_2_for, md_3_for, md_4_for, final_Ts_for, color, depth, sil, total_count, num_groups

def ol_backward(
        backward_ip,
        raster_settings, 
        md_1_back,          # pynq buffer
        md_2_back,          # pynq buffer
        md_3_back,          # pynq buffer
        means2D, 
        dL_dcoldeps,
        size,
        num_groups,
        final_Ts_back,      # pynq buffer
        md_1_for,
        depth_mask,
        color_mask
    ):
    ta = time.time()
    print("trying to : backward...")
    backward_ip.register_map.CTRL.SW_RESET = 1
    N = md_1_for.shape[0]

    # input md_1_back
    write_64bit_address(backward_ip, 0xcc, 0xd0, md_1_back.physical_address) # checked
    # input md_2_back
    write_64bit_address(backward_ip, 0xd8, 0xdc, md_2_back.physical_address) # checked
    # input md_3_back 
    write_64bit_address(backward_ip, 0xe4, 0xe8, md_3_back.physical_address) # checked
    # input final_Ts_back
    write_64bit_address(backward_ip, 0xfc, 0x100, final_Ts_back.physical_address) # checked
    
    # input depth_mask
    depth_mask = depth_mask.reshape(360*480, )
    depth_mask_buffer = allocate((360*480, ), dtype=np.bool_)
    np.copyto(depth_mask_buffer, depth_mask)
    # print("depth_mask_buffer",depth_mask_buffer)
    write_64bit_address(backward_ip, 0x108, 0x10c, depth_mask_buffer.physical_address) # checked
    
    # input color_mask
    color_mask = color_mask.reshape(360*480, )
    color_mask_buffer = allocate((360*480, ), dtype=np.bool_)
    np.copyto(color_mask_buffer, color_mask)
    # print("color_mask_buffer",color_mask_buffer)
    write_64bit_address(backward_ip, 0x114, 0x118, color_mask_buffer.physical_address) # checked

    # input dL_dcoldeps
    dL_dcoldeps = dL_dcoldeps.reshape(360*480, 4)
    dL_dcoldeps_buffer = allocate((360*480, 4), dtype=np.bool_)
    np.copyto(dL_dcoldeps_buffer, dL_dcoldeps)
    # print("\ndL_dcoldeps_buffer",dL_dcoldeps_buffer)
    write_64bit_address(backward_ip, 0xf0, 0xf4, dL_dcoldeps_buffer.physical_address) # checked

    # input focal x focal y
    focal_x = raster_settings.focal_x
    float_bytes = struct.pack('f', focal_x)
    int_value = struct.unpack('I', float_bytes)[0]
    backward_ip.write(0x22c, int_value) # checked
    focal_y = raster_settings.focal_y
    float_bytes = struct.pack('f', focal_y)
    int_value = struct.unpack('I', float_bytes)[0]
    backward_ip.write(0x234, int_value) # checked

    # input viewmatrix
    viewmatrix = raster_settings.viewmatrix.detach().numpy().reshape(-1)
    for i in range(16):  
        offset = 0x12c + i * 8 # checked
        float_bytes = struct.pack('f', viewmatrix[i])
        int_value = struct.unpack('I', float_bytes)[0]
        backward_ip.write(offset, int_value)
    # input projmatrix
    projmatrix = raster_settings.projmatrix.detach().numpy().reshape(-1)
    for i in range(16):  
        offset = 0x1ac + i * 8 # checked
        float_bytes = struct.pack('f', projmatrix[i])
        int_value = struct.unpack('I', float_bytes)[0]
        backward_ip.write(offset, int_value)

    # input num_group
    backward_ip.write(0x23c, int(num_groups)) # checked
    # input size
    backward_ip.write(0x244, int(size)) # checked
    # is forward ?
    backward_ip.write(0x24c, 0) # checked
    # is pre_grouping ?
    backward_ip.write(0x254, 0) # checked

    # md_4_back output
    md_4_back_buffer = allocate((N, 8), dtype=np.float32)
    write_64bit_address(backward_ip, 0x120, 0x124, md_4_back_buffer.physical_address) # checked

    dL_dcoldeps_buffer.flush()
    md_1_back.flush()
    md_2_back.flush()
    md_3_back.flush()
    final_Ts_back.flush()
    #print("\nall buffers ready , starting backward ip ...")
    backward_ip.write(0x00, 0x1)

    t1 = time.time()
    while True:
        if backward_ip.register_map.CTRL.AP_DONE == 1:
            break
    t2 = time.time()

    md_4_back_buffer.invalidate()

    print("buffers take :", (t1-ta)*1000, "ms")
    print("backward complete !IP total working time : ", (t2 - t1) * 1000, "ms\n")
    # print("md_4_back_buffer",md_4_back_buffer)

    return md_4_back_buffer

