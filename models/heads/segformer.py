from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..modules import MLP, ConvBNReLU
from .base import BaseHead


class SegFormerHead(BaseHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    os: int = 4

    def __init__(self,
                 in_channels: List[int],
                 in_index: Union[List[int], int],
                 num_classes: int,
                 input_transform: Optional[str] = None,
                 channels: int = 256,
                 dropout_ratio: float = 0.1,
                 ):
        super().__init__(num_classes, in_index, input_transform)
        self.in_channels = in_channels

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        embedding_dim = channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvBNReLU(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
        )

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

        # UNUSED: this is because of checkpoint loading
        # self.conv_seg = nn.Conv2d(128, self.num_classes, kernel_size=1)
        # for m in self.conv_seg.parameters():  # disable grad so it's not in optimizer
        #     m.requires_grad = False

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        # mmseg init
        nn.init.normal_(self.linear_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear_pred.bias, 0)
        for m in self.modules():
            # initialize ConvBNReLU as in mmsegmentation
            if isinstance(m, ConvBNReLU) and not m.depthwise_separable:
                nn.init.kaiming_normal_(
                    m.conv.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m.conv, 'bias') and m.conv.bias is not None:
                    nn.init.constant_(m.conv.bias, 0)
                if m.use_norm:
                    if hasattr(m.bn, 'weight') and m.bn.weight is not None:
                        nn.init.constant_(m.bn.weight, 1)
                    if hasattr(m.bn, 'bias') and m.bn.bias is not None:
                        nn.init.constant_(m.bn.bias, 0)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32

        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3])
        _c4 = nn.functional.interpolate(
            _c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.functional.interpolate(
            _c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3])
        _c2 = nn.functional.interpolate(
            _c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3])

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.linear_pred(x)

        return x
