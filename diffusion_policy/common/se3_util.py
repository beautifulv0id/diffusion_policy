import torch

def se3_from_rot_pos(rot, pos):
    """
    Convert rotations and positions to SE(3) matrices.

    Args:
        rot: rotations as rotation matrices, as tensor of shape (..., 3, 3).
        pos: positions as translation vectors, as tensor of shape (..., 3).

    Returns:
        SE(3) matrices as tensor of shape (..., 4, 4).
    """
    H = torch.eye(4, device=rot.device, dtype=rot.dtype).expand(rot.shape[:-2] + (4, 4)).clone()
    H[..., :3, :3] = rot
    H[..., :3, 3] = pos
    return H

def rot_pos_from_se3(H):
    """
    Convert SE(3) matrices to rotations and positions.

    Args:
        H: SE(3) matrices, as tensor of shape (..., 4, 4).

    Returns:
        Tuple of rotations as rotation matrices, as tensor of shape (..., 3, 3),
        and positions as translation vectors, as tensor of shape (..., 3).
    """
    return H[..., :3, :3], H[..., :3, 3]

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    dtype = torch.float32
    rot = torch.randn(2, 3, 3, device=device, dtype=dtype)
    pos = torch.randn(2, 3, device=device, dtype=dtype)
    se3 = se3_from_rot_pos(rot, pos)
    assert se3.shape == (2, 4, 4)
    assert se3.device == rot.device
    assert se3.dtype == rot.dtype
    assert torch.allclose(se3[..., :3, :3], rot)
    assert torch.allclose(se3[..., :3, 3], pos)


if __name__ == "__main__":
    test()
    print("Passed")