import torch
import torch.nn.functional as F
import numpy as np
import skimage
import skimage.transform
import torch.hub

def get_dino_features(img_raw:np.ndarray, scale:int=1)->torch.Tensor:
    """get dino features for only one img

    Args:
        img (np.ndarray): (h, w, 3)
        scale (int, optional): _description_. Defaults to 3.

    Returns:
        torch.Tensor: (h, w, F)
    """
    assert np.all(img_raw <= 1.0)

    img_raw = img_raw.astype('float32')
    img_raw = skimage.img_as_float32(img_raw)

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').cuda()

    h, w = img_raw.shape[0] // 14 * 14,  img_raw.shape[1] // 14 * 14
    img = skimage.transform.resize(
                img_raw,
                (h,w)
            ).astype('float32')
    img = torch.from_numpy(img)
    img = img[None, :].cuda()
    # print('Picture for Dino size:', img.shape)
    with torch.no_grad():
        ### (batch size, 3, height, width)
        ret = model.forward_features(img.permute(0, 3, 1, 2))
        #ret = model.forward_features(img.permute(0, 3, 1, 2))
    features = ret['x_norm_patchtokens']
    N, _, F = features.shape

    ### (height, width, features) torch.Tensor np.float32
    features = features.reshape(img_raw.shape[0]//14, img_raw.shape[1]//14, F).permute(2, 0, 1)
    features = torch.nn.functional.interpolate(features.unsqueeze(0), size=(img_raw.shape[0] // scale , img_raw.shape[1] // scale ), mode='bilinear', align_corners=False)
    features = features.squeeze(0).permute(1, 2, 0)

    return features