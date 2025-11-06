import torch
import numpy as np
from utils.rasterize_helpers import transformed_params2rendervar
from utils.rasterization import GaussianRasterizer as Renderer

def method_custom_equal_height(depth_map: np.ndarray, n_bins: int = 256) -> np.ndarray:
    depths = depth_map.flatten()
    hist, bin_edges = np.histogram(depths, bins=4096)

    total_pixels = len(depths)
    pixels_per_bin = total_pixels // n_bins

    boundaries = []
    cumulative_pixels = 0
    current_bin_pixels = 0
    for i in range(len(hist)):
        cumulative_pixels += hist[i]
        current_bin_pixels += hist[i]

        if cumulative_pixels >= pixels_per_bin * (len(boundaries) + 1):
            if i > 0 and hist[i] > 0:
                fraction = (pixels_per_bin * (len(boundaries) + 1) -
                            (cumulative_pixels - hist[i])) / hist[i]
                boundary_value = bin_edges[i] + fraction * (bin_edges[i + 1] - bin_edges[i])
            else:
                boundary_value = bin_edges[i + 1]
            boundaries.append(boundary_value)

    boundaries = boundaries[:255]
    boundaries.append(bin_edges[-1])
    return np.array(boundaries)

# def method_custom_equal_height(depth_map: np.ndarray, n_bins: int = 64) -> np.ndarray:
#     depths = depth_map.flatten()
#     hist, bin_edges = np.histogram(depths, bins=1024)

#     total_pixels = len(depths)
#     pixels_per_bin = total_pixels // n_bins

#     boundaries = []
#     cumulative_pixels = 0
#     current_bin_pixels = 0
#     for i in range(len(hist)):
#         cumulative_pixels += hist[i]
#         current_bin_pixels += hist[i]

#         if cumulative_pixels >= pixels_per_bin * (len(boundaries) + 1):
#             if i > 0 and hist[i] > 0:
#                 fraction = (pixels_per_bin * (len(boundaries) + 1) -
#                             (cumulative_pixels - hist[i])) / hist[i]
#                 boundary_value = bin_edges[i] + fraction * (bin_edges[i + 1] - bin_edges[i])
#             else:
#                 boundary_value = bin_edges[i + 1]
#             boundaries.append(boundary_value)

#     boundaries = boundaries[:63]
#     boundaries.append(bin_edges[-1])
#     return np.array(boundaries)


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True,
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[1], color.shape[0]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).float(),
                                    torch.arange(height).float(),
                                    indexing='xy')
    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Colorize point cloud
    cols = color.reshape(-1, 3)
    point_cld = torch.cat((pts, cols), -1)

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist and mean_sq_dist_method == "projective":
        mean3_dist = depth_z / ((FX + FY) / 2)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_dist = mean3_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_dist
    else:
        return point_cld

def initialize_params(init_pt_cld, num_frames, mean3_dist):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cpu")
    scales = torch.tile(mean3_dist[..., None], (1, 1))

    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(mean3_dist)[..., None], (1, 1)),
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).float()}

    return params, variables

def initialize_new_params(new_pt_cld, mean3_dist):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cpu")
    scales = torch.tile(mean3_dist[..., None], (1, 1))
    
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(mean3_dist)[..., None], (1, 1)),
    }

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cpu().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cpu().float().contiguous().requires_grad_(True))

    return params 

def add_new_gaussians(ip, params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method):

    rendervar = transformed_params2rendervar(params)
    _, render_depth, silhouette = Renderer(ip=ip, raster_settings=curr_data['cam'])(**rendervar) 

    non_presence_sil_mask = (silhouette < sil_thres)

    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth']
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)                             # (H, W, 1)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median()) # (H, W, 1)
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask                           # (H, W, 1)

    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    print("\n\n\nnon_presence_sum",torch.sum(non_presence_mask))
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_w2c = curr_data['curr_w2c']
        valid_depth_mask = (curr_data['depth'] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        
        new_pt_cld, mean3_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method) 
        print("new_pt_cld",new_pt_cld)
        new_params = initialize_new_params(new_pt_cld, mean3_dist)

        for k, v in new_params.items(): 
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
       
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cpu").float()
        variables['denom'] = torch.zeros(num_pts, device="cpu").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cpu").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cpu").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables

def remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys()]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    if 'timestep' in variables.keys():
        variables['timestep'] = variables['timestep'][to_keep]
    return params, variables

def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def prune_gaussians(params, variables, optimizer, iter, prune_dict):
    if iter <= prune_dict['stop_after']: 
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            
            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).unsqueeze(-1).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            
            params, variables = remove_points(to_remove, params, variables, optimizer)
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']: 
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
    
    return params, variables

def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables

def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params