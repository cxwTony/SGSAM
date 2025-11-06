import torch
from utils.rasterization import GaussianRasterizationSettings as Camera

def setup_camera(w, h, thresholds, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).float()
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        focal_x=fx,
        focal_y=fy,
        viewmatrix=w2c,
        projmatrix=full_proj,
        thresholds=thresholds
    )
    return cam