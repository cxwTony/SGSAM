import torch
import torch.nn.functional as F
from utils.rasterization import GaussianRasterizer as Renderer
from utils.common_helpers import *
from pynq.lib.video import *
import cv2
import time

def transformed_params2rendervar(params):
    # Initialize Render Variables
    opacities = torch.sigmoid(params['logit_opacities'])
    raddi = torch.exp(params['log_scales'])
    rendervar = {
        'md_1_for': torch.cat([params['means3D'], raddi, params['rgb_colors'], opacities ], dim=1).contiguous(),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cpu") + 0,
    }
    return rendervar

def get_loss(ip, params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, do_ba=False):
    # Initialize Loss Dictionary & Render Variables
    losses = {}
    tr1 = time.time()
    rendervar = transformed_params2rendervar(params)
    tr2 = time.time()
    
    
    # RGB Rendering
    renders = time.time()
    rendervar['means2D'] = rendervar['means2D'].detach()
    im, depth, silhouette = Renderer(ip, raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    rendere = time.time()

    tl1 = time.time()
    # Mask with valid depth values (accounts for outlier depth values)
    presence_sil_mask = (silhouette > sil_thres)
    nan_mask = (~torch.isnan(depth))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10 * depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    mask = mask & presence_sil_mask

    cmask = presence_sil_mask
    cmask = torch.tile(cmask, (1, 1, 3))
    
    # Depth loss
    if use_l1:
        mask = mask.detach()
        losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    # RGB Loss
    if use_l1:
        cmask = cmask.detach()
        losses['im'] = torch.abs(im - curr_data['im'])[cmask].mean()
    
    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
                             
    loss = sum(weighted_losses.values())
    tl2 = time.time()
    print("transformed_params2rendervar time = ", (tr2-tr1)*1000,"ms")
    print("render time = ", (rendere-renders)*1000,"ms")
    print("cal loss time = ", (tl2-tl1)*1000,"ms")
    print(" iter_time_idx = ",iter_time_idx,"\n closs = ",losses['im'],"\n dloss = ", losses['depth'],"\n")
    
    # seen = radius > 0
    # variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    # variables['seen'] = seen
    weighted_losses['loss'] = loss
    

    return loss, variables, weighted_losses, im