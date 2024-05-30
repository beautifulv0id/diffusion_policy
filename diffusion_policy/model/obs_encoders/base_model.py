import torch
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class BaseObservationEncoder(ModuleAttrMixin):
    def __init__(self):
        super(BaseObservationEncoder, self).__init__()

    def forward(self, context):
        return context

    def get_args(self):
        return {
            '__class__': [type(self).__module__, type(self).__name__]
        }