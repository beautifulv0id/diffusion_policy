from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import clip
import einops

def load_clip():
    clip_model, clip_transforms = clip.load("RN50")
    normalize = clip_transforms.transforms[-1]
    return clip_model, normalize

class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            n_obs_steps, 
            action_dim, 
            global_dim,
            img_shape: Tuple[int,int]=(128,128),
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()


        # handle sharing vision backbone
        self.clip, self.normalize = load_clip()
        for p in self.clip.parameters():
            p.requires_grad = False


        h, w = img_shape
        randomizer = nn.Identity()
        if crop_shape is not None:
            ch, cw = crop_shape
            if random_crop:
                randomizer = CropRandomizer(
                    input_shape=(3,h,w),
                    crop_height=ch,
                    crop_width=cw,
                    num_crops=1,
                    pos_enc=False
                )
            else:
                randomizer = torchvision.transforms.CenterCrop(
                    size=(ch,cw)
                )
        resizer = torchvision.transforms.Resize(
            size=(224,224)
        )
        self.transform = nn.Sequential(randomizer, resizer)

        self.clip_out_proj = nn.Linear(1024, global_dim)

        self.gripper_embeddings = nn.Embedding(n_obs_steps, global_dim)
        self.gripper_head = nn.Sequential(
            nn.Linear(action_dim, global_dim),
            nn.ReLU()
        )


    def forward(self, curr_gripper, rgb):
        b, v = rgb.shape[:2]
        rgb = einops.rearrange(rgb, 'b v c h w -> (b v) c h w')
        rgb = self.transform(rgb)
        rgb = self.normalize(rgb)
        rgb_features = self.clip.encode_image(rgb).float()
        rgb_features = self.clip_out_proj(rgb_features)
        rgb_features = einops.rearrange(rgb_features, '(b v) d -> b v d', b=b)
        gripper_features = self.gripper_head(curr_gripper) + self.gripper_embeddings.weight.unsqueeze(0).repeat(b, 1, 1)

        features = torch.cat([rgb_features, gripper_features], dim=1)
        return features
    
        

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MultiImageObsEncoder(
        n_obs_steps=3,
        action_dim=10,
        global_dim=256,
        crop_shape=(75,75),
        random_crop=True
    )
    encoder.to(device)
    curr_gripper = torch.randn(1, 3, 10).to(device)
    rgb = torch.randn(1, 4, 3, 128, 128).to(device)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    example_output = encoder.forward(curr_gripper, rgb)
    print(example_output.shape)

if __name__ == '__main__':  
    main()