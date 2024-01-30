import torch
import torchvision

import clip
from clip.model import ModifiedResNet

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model


def get_clip(features=None):
    clip_model, clip_transforms = clip.load("RN50")
    state_dict = clip_model.state_dict()
    layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                    for b in [1, 2, 3, 4]])
    output_dim = state_dict["text_projection"].shape[1]
    heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
    backbone = ModifiedResNetFeatures(layers, output_dim, heads, features=features)
    backbone.load_state_dict(clip_model.visual.state_dict())
    normalize = clip_transforms.transforms[-1]
    return torch.nn.Sequential(normalize, backbone)


class ModifiedResNetFeatures(ModifiedResNet):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, features=None):
        super().__init__(layers, output_dim, heads, input_resolution, width)
        if isinstance(features, str):
            if features == "all":
                features = ["res1", "res2", "res3", "res4", "res5"]
            else:
                features = [features]
        self.features = features
    def forward(self, x: torch.Tensor):
        out = dict()
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))

        if "res1" in self.features:
            out["res1"] =  x0
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        if "res2" in self.features:
            out["res2"] =  x1
        x2 = self.layer2(x1)
        if "res3" in self.features:
            out["res3"] =  x2
        x3 = self.layer3(x2)
        if "res4" in self.features:
            out["res4"] =  x3
        x4 = self.layer4(x3)
        if "res5" in self.features:
            out["res5"] =  x4

        return out