import torch
import torch.nn.functional as F

def rotate(x: torch.Tensor, angle: torch.Tensor, mode='bilinear', padding_mode='border') -> torch.Tensor:
    """
    Rotate batch of images [B, C, W, H] by a specific angles [B] 
    (counter clockwise)

    Parameters
    ----------
    x : torch.Tensor
      batch of images
    angle : torch.Tensor
      batch of angles

    Returns
    -------
    torch.Tensor
        batch of rotated images
    """
    if isinstance(angle, int) or isinstance(angle, float):
      angle = torch.tensor([angle], device=x.device)

    h, w = x.shape[-2:]
    s = torch.sin(angle)
    c = torch.cos(angle)
    rot_mat = torch.stack((torch.stack([c, -s], dim=1),
                           torch.stack([s, c], dim=1)), dim=1)
    zeros = torch.zeros(rot_mat.size(0), 2).unsqueeze(2).to(x.device)
    aff_mat = torch.cat((rot_mat, zeros), 2)
    aff_mat[:,:,0] *= w / 2
    aff_mat[:,:,1] *= h / 2
    grid = F.affine_grid(aff_mat, x.size(), align_corners=True)
    grid[...,0] /= w / 2
    grid[...,1] /= h / 2
    x = F.grid_sample(x, grid, align_corners=True, mode=mode, padding_mode=padding_mode)
    return x