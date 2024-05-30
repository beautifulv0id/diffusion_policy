import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

class AdaptAndNormalize(ModuleAttrMixin):

    def __init__(self, normalizer=None, adaptor=None):
        super(AdaptAndNormalize, self).__init__()

        self.normalizer = normalizer
        self.adaptor = adaptor

    def normalize(self, data):

        if self.adaptor is not None:
            data = self.adaptor.adapt(data)
        if self.normalizer is not None:
            data = self.normalizer.normalize(data)
        return data

    def unnormalize(self, data):
        if self.normalizer is not None:
            data = self.normalizer.unnormalize(data)
        if self.adaptor is not None:
            data = self.adaptor.unadapt(data)
        return data



