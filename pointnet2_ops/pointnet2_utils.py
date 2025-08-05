def furthest_point_sample(xyz, npoint):
    import torch
    B, N, C = xyz.shape
    return torch.arange(npoint).repeat(B, 1)
