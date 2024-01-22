import torch
import torch.nn as nn
import torchvision

class CenterCrop(torchvision.transforms.CenterCrop):
    def __init__(self, size, pos_enc=False):
        super().__init__(size)
        self.pos_enc = pos_enc
        print(f"CenterCrop: {size} with pos_enc={pos_enc}"  )

    def forward(self, x):
        src_shape = torch.tensor(x.shape[-2:])
        out = super().forward(x)
        if self.pos_enc:
            out_shape = torch.tensor(out.shape[-2:])
            h, w = out.shape[-2:]
            pos_y, pos_x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w))
            pos_y = pos_y.float().to(out.device)
            pos_x = pos_x.float().to(out.device)
            position_enc = torch.stack((pos_y, pos_x), dim=0)
            offset = ((1 - out_shape / src_shape) / 2).reshape(2, 1, 1)
            scale = (out_shape / src_shape).reshape(2, 1, 1)
            position_enc = (position_enc * scale + offset).unsqueeze(0)
            out = torch.cat((out, position_enc), dim=1)
        return out
