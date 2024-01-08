import torch
from torch import nn
import warnings
from torch.nn import functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class GFM(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, kernel_size=1, stride=1, padding=0):
        super(GFM, self).__init__()
        self.out_channels = out_channels

        self.gate_conv1_1 = nn.Conv2d(in_channels1, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding)
        self.gate_conv2_1 = nn.Conv2d(in_channels2, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding)
        self.gate_conv1_2 = nn.Conv2d(in_channels1, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding)
        self.gate_conv2_2 = nn.Conv2d(in_channels2, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x1, x2):
        s1 = self.gate_conv1_1(x1)
        g1 = self.gate_conv1_2(x1)

        s2 = self.gate_conv2_1(x2)
        g2 = self.gate_conv2_2(x2)

        res1 = s1 * torch.sigmoid(g2) + s1
        res2 = s2 * torch.sigmoid(g1) + s2
        res = torch.cat([res1, res2], dim=1)
        return self.conv_out(res)


class Decoder(nn.Module):
    def __init__(self, embedding_dim=64, channels=None):
        super(Decoder, self).__init__()
        if channels is None:
            channels = [64, 128, 320, 512]
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = channels
        self.fuse1 = GFM(c1_in_channels, embedding_dim, embedding_dim)
        self.fuse2 = GFM(c2_in_channels, embedding_dim, embedding_dim)
        self.fuse3 = GFM(c3_in_channels, c4_in_channels, embedding_dim)

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        fuses = []
        _c3 = resize(c4, size=c3.size()[2:], mode='bilinear', align_corners=False)
        fuse = self.fuse3(c3, _c3)
        fuses.append(fuse)
        _c2 = resize(fuse, size=c2.size()[2:], mode='bilinear', align_corners=False)
        fuse = self.fuse2(c2, _c2)
        fuses.append(fuse)
        _c1 = resize(fuse, size=c1.size()[2:], mode='bilinear', align_corners=False)
        fuse = self.fuse1(c1, _c1)
        fuses.append(fuse)
        return fuses
