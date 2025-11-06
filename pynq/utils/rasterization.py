from typing import NamedTuple
import cv2
import numpy as np
import torch.nn as nn
import torch
import time
import pynq

from utils.ip_helpers import *

def create_neighbor_tensor(color, depth):
    combined = torch.cat([color, depth], dim=2).contiguous().detach().numpy()  # shape: (360,480,4)
    bool_coldepth = ( combined > 0 )
    return bool_coldepth

def rasterize_gaussians(
    md_1_for,
    means2D,
    ip,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        md_1_for,
        means2D,
        ip,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        md_1_for,
        means2D,
        ip,
        raster_settings,
    ):
        
        t_ol_1 = time.time()
        md_2_for, md_3_for, md_4_for, final_Ts_for, color, depth, sil, total_count, num_groups = ol_forward(
                md_1_for,
                means2D,
                ip,
                raster_settings,
        )
        t_ol_2 = time.time()
        print("time ol_forward ",1000*(t_ol_2-t_ol_1),"ms")

        # Keep relevant tensors for backward
        ctx.ip = ip
        ctx.raster_settings = raster_settings
        ctx.md_1_back = md_2_for
        ctx.md_2_back = md_3_for
        ctx.md_3_back = md_4_for
        ctx.size = total_count
        ctx.num_groups = num_groups
        ctx.final_Ts_for = final_Ts_for

        ctx.save_for_backward(md_1_for, means2D)
        
        color = color.reshape(360, 480, 3)
        depth = depth.reshape(360, 480, 1)
        sil = sil.reshape(360, 480, 1)
        color = torch.from_numpy(color).float().contiguous().requires_grad_(True)
        depth = torch.from_numpy(depth).float().contiguous().requires_grad_(True)
        sil = torch.from_numpy(sil).float().contiguous().requires_grad_(True)
        
        return color, depth, sil

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, _):

        # # Restore necessary values from context
        t = time.time()
        ip = ctx.ip
        md_1_back = ctx.md_1_back
        md_2_back = ctx.md_2_back
        md_3_back = ctx.md_3_back
        size = ctx.size
        raster_settings = ctx.raster_settings
        num_groups = ctx.num_groups
        final_Ts_back = ctx.final_Ts_for
        
        dL_dcoldeps = create_neighbor_tensor(grad_out_color, grad_out_depth)
        
        md_1_for, means2D = ctx.saved_tensors
#         print("grad_out_color",grad_out_color)
#         print("grad_out_depth",grad_out_depth)
        depth_mask = (grad_out_depth == 0)
        color_mask = (grad_out_color[:, :, 0] == 0)
#         depth_mask = (grad_out_depth != 20000)
#         color_mask = (grad_out_color[:, :, 0] != 20000)
        tt = time.time()
        print("before backward",1000*(tt-t),"ms")

        md_4_back = ol_backward(
                            ip,
                            raster_settings, 
                            md_1_back, 
                            md_2_back, 
                            md_3_back, 
                            means2D, 
                            dL_dcoldeps,
                            size,
                            num_groups,
                            final_Ts_back,
                            md_1_for,
                            depth_mask,
                            color_mask
                            )
        
        grad_md_4_back = torch.from_numpy(md_4_back).float().contiguous()
        
        print("ol_backward",1000*(time.time()-tt),"ms")
        
        grads = (
            grad_md_4_back,
            None,
            None,
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    focal_x : float
    focal_y : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    thresholds : np.ndarray


class GaussianRasterizer(nn.Module):
    def __init__(self, ip, raster_settings):
        super().__init__()
        self.ip = ip
        self.raster_settings = raster_settings

    def forward(self, md_1_for, means2D):
        ip = self.ip
        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            md_1_for,
            means2D,
            ip,
            raster_settings,
        )