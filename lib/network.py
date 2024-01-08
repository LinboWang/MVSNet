import torch
from torch import nn
from lib.decoder import Decoder
from lib.encoder import Encoder
from torch.nn import functional as F
from torchvision import transforms


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class MultiViewAggregation(nn.Module):
    def __init__(self, feat_chnl=512, embed_chnl=512, views=3):
        super(MultiViewAggregation, self).__init__()
        self.conv1 = BasicConv2d(in_planes=feat_chnl * views, out_planes=embed_chnl, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=embed_chnl, out_channels=views, kernel_size=1)

    def forward(self, feats, preds, ops):
        op_feats = []
        op_preds = []
        for i, op in enumerate(ops):
            op_feats.append(op(feats[i]))
            op_preds.append(op(preds[i]))
        op_feat = torch.cat(op_feats, dim=1)
        op_pred = torch.cat(op_preds, dim=1)
        op_feat = self.conv1(op_feat)
        wgts = self.conv2(op_feat)
        wgts = F.softmax(wgts)
        out = torch.sum(op_pred * wgts, dim=1, keepdim=True)
        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)


class MVSNet(nn.Module):
    def __init__(self, emb_dim=64):
        super(MVSNet, self).__init__()
        self.horizontal = transforms.RandomHorizontalFlip(p=1)
        self.diagonal = transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomHorizontalFlip(p=1)
        ])
        self.normal = transforms.Compose([])
        self.former_encoder = Encoder()
        self.ops = [self.horizontal, self.normal, self.diagonal]
        for i in range(0, len(self.ops)):
            former_decoder = Decoder(embedding_dim=emb_dim)
            linear_pred = nn.Conv2d(in_channels=emb_dim, out_channels=1, kernel_size=1)
            setattr(self, "former_decoder{}".format(i), former_decoder)
            setattr(self, "linear_pred{}".format(i), linear_pred)
        self.mv_fuse = MultiViewAggregation(embed_chnl=emb_dim, feat_chnl=emb_dim, views=len(self.ops))

    def forward(self, x, gts=None):
        decoder_feats = []
        pred_outs = []
        mask_outs = []
        for i, op in enumerate([self.horizontal, self.normal, self.diagonal]):
            former_decoder = getattr(self, "former_decoder{}".format(i))
            linear_pred = getattr(self, "linear_pred{}".format(i))
            op_x = op(x)
            encoder_outs = self.former_encoder(op_x)
            decoder_outs = former_decoder(encoder_outs)
            preds = [linear_pred(decoder_out) for decoder_out in decoder_outs]
            mask = [F.interpolate(pred, size=x.size()[2:], mode='bilinear', align_corners=True) for pred in preds]
            decoder_feats.append(decoder_outs[-1])
            pred_outs.append(preds[-1])
            mask_outs.append(mask)
        mv_mask = self.mv_fuse(decoder_feats, pred_outs, self.ops)
        if gts is not None:
            losses = 0
            for i, op in enumerate([self.horizontal, self.normal, self.diagonal]):
                op_gt = op(gts)
                losses += structure_loss(mask_outs[i][0], op_gt)
                losses += structure_loss(mask_outs[i][1], op_gt)
                losses += structure_loss(mask_outs[i][2], op_gt)
            losses += structure_loss(mv_mask, gts)
            return losses
        else:
            return mv_mask
