from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Normalize as Normalize
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.vision.center_crop import CenterCrop
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding2D, SinusoidalPosEmb
from diffusion_policy.model.common.layer import RelativeCrossAttentionModule
import einops
from torchvision.ops import FeaturePyramidNetwork

class TransformerHybridObsRelativeEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            feature_pyramid: FeaturePyramidNetwork,
            feature_map_pyramid: list,
            n_obs_steps: int,
            query_embeddings: nn.Embedding,
            rotary_embedder: RotaryPositionEncoding2D,
            positional_embedder: SinusoidalPosEmb,
            within_attn : RelativeCrossAttentionModule,
            across_attn : RelativeCrossAttentionModule,
            rgb_model_frozen: bool=True,
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            # renormalize rgb input with clip normalization
            clip_norm: bool=False,
        ):
        """
        Assumes rgb input: B,To,C,H,W
        Assumes low_dim input: B,To,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_normalizer_map = nn.ModuleDict()
        key_shape_map = dict()
        key_feature_pyramid_map = nn.ModuleDict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module), \
                "Must provide nn.Module if sharing rgb model"
            key_model_map['rgb'] = rgb_model
            if feature_pyramid is not None:
                assert isinstance(feature_pyramid, nn.Module), \
                    "Must provide nn.Module if sharing rgb model"
                key_feature_pyramid_map['rgb'] = feature_pyramid
            else:
                key_feature_pyramid_map['rgb'] = nn.Identity()

        obs_shape_meta = shape_meta['obs']
        assert ("agent_pos" in obs_shape_meta), "Must have agent_pos in obs_shape_meta"
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = (n_obs_steps,) + shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module), \
                            "Must provide nn.Module if not sharing rgb model"
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )

                    key_model_map[key] = this_model
                    if feature_pyramid is not None:
                        this_feature_pyramid = None
                        if isinstance(feature_pyramid, dict):
                            # have provided model for each key
                            this_feature_pyramid = feature_pyramid[key]
                        else:
                            assert isinstance(feature_pyramid, nn.Module), \
                                "Must provide nn.Module if not sharing rgb model"
                            # have a copy of the rgb model
                            this_feature_pyramid = copy.deepcopy(feature_pyramid)
                        key_feature_pyramid_map[key] = this_feature_pyramid
                    else:
                        key_feature_pyramid_map[key] = nn.Identity()
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=True
                        )
                    else:
                        this_normalizer = CenterCrop(
                            size=(h,w),
                            pos_enc=True
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                elif clip_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073], 
                        std=[0.26862954, 0.26130258, 0.27577711])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer)
                key_transform_map[key] = this_transform
                key_normalizer_map[key] = this_normalizer
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
    
        if rgb_model_frozen:
            for m in key_model_map.values():
                m.eval()
                m.requires_grad_(False)     

        key_pyramid_embedding_map = nn.ModuleDict()
        for key in feature_map_pyramid:
            key_pyramid_embedding_map[key] = nn.Embedding(
                num_embeddings=1,
                embedding_dim=query_embeddings.embedding_dim
            )               

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_normalizer_map = key_normalizer_map
        self.key_feature_pyramid_map = key_feature_pyramid_map
        self.feature_map_pyramid = feature_map_pyramid
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.rgb_model_frozen = rgb_model_frozen
        self.rotary_embedder = rotary_embedder
        self.query_emb = query_embeddings
        self.key_pyramid_embedding_map = key_pyramid_embedding_map
        self.re_cross_attn_within = within_attn
        self.re_cross_attn_across = across_attn
        self.positional_embedder = positional_embedder

    def rotary_embed(self, x):
        """
        Args:
            x (torch.Tensor): (B, N, Da)
        Returns:
            torch.Tensor: (B, N, D, 2)
        """
        return self.rotary_embedder(x)
    
    def compute_visual_features(self, obs_dict):
        batch_size = None
        img_positions = list()
        key_img_features_map = {key: list() for key in self.feature_map_pyramid}
        key_img_positions_map = {key: None for key in self.feature_map_pyramid}
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                if self.training and self.crop_shape is not None:
                    img, pos = img[...,:3,:,:], img[...,3:,:,:]
                    img_positions.append(pos)
                img = self.key_normalizer_map[key](img)
                imgs.append(img)
            # (B,N,C,H,W)
            imgs = torch.cat(imgs, dim=1) # TODO: check if cat along dim 1 is correct, probabliy need to unsqueeze dim 1
            # (N*B,C,H,W)
            imgs = imgs.reshape(-1,*imgs.shape[2:])
            # (B*N,D,H,W) TODO: check if this is correct previous was (N*B,D,H,W)
            with torch.no_grad() if self.rgb_model_frozen else torch.enable_grad():
                feature = self.key_model_map['rgb'](imgs)
            # {f:(B*N,D,H,W)}
            feature = self.key_feature_pyramid_map['rgb'](feature)
            # {f:(B,N,D,H,W)}
            feature = dict_apply(feature, lambda f: f.reshape(batch_size, self.n_obs_steps, *f.shape[1:]))
            for k, v in feature.items():
                v = v + self.key_pyramid_embedding_map[k].weight.reshape((1,1,-1,1,1))
                key_img_features_map[k].append(v)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                if self.training and self.crop_shape is not None:
                    img, pos = img[...,:3,:,:], img[...,3:,:,:]
                    img_positions.append(pos)
                img = self.key_normalizer_map[key](img)
                with torch.no_grad() if self.rgb_model_frozen else torch.enable_grad():
                    feature = self.key_model_map[key](img)
                feature = self.key_feature_pyramid_map[key](feature)
                for k, v in feature.items():
                    v = v + self.key_pyramid_embedding_map[k].weight.reshape((1,1,-1,1,1))
                    key_img_features_map[k].append(v)
        
        # TODO: check if cat along dim 1 is correct for share_rgb_model=False
        key_img_features_map = dict_apply(key_img_features_map, lambda x: torch.cat(x, dim=1))

        if self.training and self.crop_shape is not None:
            # (B,N,C,H,W)
            img_positions = torch.cat(img_positions, dim=1)
            for k, img_feat in key_img_features_map.items():
                positions = torch.nn.functional.interpolate(img_positions.flatten(end_dim=1), size=img_feat.shape[-2:], mode='bilinear', align_corners=False) # TODO: check if this is correct
                positions = einops.rearrange(positions, '(b n) d h w -> b n d h w', b=batch_size)
                key_img_positions_map[k] = img_positions
        else:
            for k, img_feat in key_img_features_map.items():
                img_positions = self.position_enc_from_shape(img_feat.shape[-2:])
                img_positions = pos = einops.repeat(img_positions, 'c h w -> b n c h w', b=batch_size, n=self.n_obs_steps)
                key_img_positions_map[k] = img_positions
        
        key_img_positions_map = dict_apply(key_img_positions_map, lambda x: x * 2. - 1.)

        key_img_positions_map = dict_apply(key_img_positions_map, lambda x: x.flatten(start_dim=-2))
        key_img_features_map = dict_apply(key_img_features_map, lambda x: x.flatten(start_dim=-2))

        img_positions = torch.cat(list(key_img_positions_map.values()), dim=-1)
        img_features = torch.cat(list(key_img_features_map.values()), dim=-1)

        return img_features, img_positions
    
    def process_low_dim_features(self, obs_dict):
        batch_size = None
        low_dim_features = list()
        agent_pos : torch.Tensor
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            if key == 'agent_pos':
                agent_pos = data
            else:
                low_dim_features.append(data)
        if len(low_dim_features) > 0:
            low_dim_features = torch.cat(low_dim_features, dim=1)
        return agent_pos, low_dim_features
    
    def position_enc_from_shape(self, shape):
        h, w = shape
        pos_y, pos_x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w))
        pos_y = pos_y.float().to(self.device)
        pos_x = pos_x.float().to(self.device)
        position_enc = torch.stack((pos_x, pos_y), dim=0)
        return position_enc

    def forward(self, obs_dict):
        batch_size = obs_dict[self.low_dim_keys[0]].shape[0]
        N = self.n_obs_steps

        # process rgb input
        # (B,N,D,M)
        img_features, img_positions = self.compute_visual_features(obs_dict)
        
        # process lowdim input
        # (B,N,D), (B,N,D,H,W)
        agent_pos, low_dim_features = self.process_low_dim_features(obs_dict) # TODO: make use of other low_dim_features?

        # compoute context features
        # (H*W,B*N,D)
        context_features = einops.rearrange(img_features, 'b n d hw -> hw (b n) d')
        # (B*N,H*W,D)
        context_pos = einops.repeat(img_positions, 'b n d hw -> (b n) hw d')
        # (B*N,H*W,D,2)
        context_pos = self.rotary_embed(context_pos)

        # compute query features
        # (1,B*N,D)
        query = einops.repeat(self.query_emb.weight, "n d -> 1 (b n) d", b=batch_size)
        # (B*N,1,D)
        query_pos = einops.repeat(agent_pos, "b n d -> (b n) 1 d")
        # (B*N,1,D,2)
        query_pos = self.rotary_embed(query_pos).type(query.dtype)

        # cross attention within observation
        # (L,B*N,D)
        obs_embs = self.re_cross_attn_within(query=query, value=context_features, 
                                             query_pos=query_pos, value_pos=context_pos)
        # (B*N,D)
        obs_embs = obs_embs[-1].squeeze(0)
        # (N,B,D)
        obs_embs = einops.rearrange(obs_embs, '(b n) d -> n b d', b=batch_size)
        # (N,1,D)
        obs_pos_embs = self.positional_embedder(torch.arange(self.n_obs_steps, dtype=obs_embs.dtype, device=obs_embs.device)).unsqueeze(1)
        # (N,B,D)
        obs_embs = obs_embs + obs_pos_embs

        # cross attention across observations
        obs_emb = self.re_cross_attn_across(query=obs_embs[-1:], 
                                            value=obs_embs[:-1])[0]\
                                                .squeeze(0)

        result = obs_emb
        return result
    
    @torch.no_grad()
    def visual_output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        To = self.n_obs_steps
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,To) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output, _ = self.compute_visual_features(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

    
    @torch.no_grad()
    def rgb_output_shapes(self):
        if self.share_rgb_model:
            example_in = torch.zeros(
                (1,) + self.key_shape_map['rgb'], 
                dtype=self.dtype,
                device=self.device)
        example_out = self.key_model_map['rgb'](example_in)
        output_shapes = dict()
        for k, v in example_out.items():
            output_shapes[k] = v.shape[1:]
        return output_shapes

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        To = self.n_obs_steps
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,To) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape


