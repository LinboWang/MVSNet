import torch
from torch import nn
from lib.pvtv2 import pvt_v2_b2


def stage_forward(x, patch_embed, block, norm):
    B = x.shape[0]
    x, H, W = patch_embed(x)
    for i, blk in enumerate(block):
        x = blk(x, H, W)
    x = norm(x)
    x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = pvt_v2_b2()
        path = r'./pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if
                      (k in model_dict.keys())}

        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        outs = []
        x = stage_forward(x, self.backbone.patch_embed1, self.backbone.block1, self.backbone.norm1)
        outs.append(x)
        x = stage_forward(x, self.backbone.patch_embed2, self.backbone.block2, self.backbone.norm2)
        outs.append(x)
        x = stage_forward(x, self.backbone.patch_embed3, self.backbone.block3, self.backbone.norm3)
        outs.append(x)
        x = stage_forward(x, self.backbone.patch_embed4, self.backbone.block4, self.backbone.norm4)
        outs.append(x)
        return outs
