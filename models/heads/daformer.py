from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..modules import MLP, ConvBNReLU
from .base import BaseHead


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, norm_layer, activation_layer):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        for dilation in dilations:
            self.append(
                ConvBNReLU(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    norm_layer=self.norm_layer,
                    activation_layer=self.activation_layer))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = ConvBNReLU(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_layer=self.norm_layer,
                    activation_layer=self.activation_layer,
                    depthwise_separable=True)


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_layer,
                 activation_layer,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvBNReLU(
                    in_channels,
                    channels,
                    1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer)
        self.bottleneck = ConvBNReLU(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                nn.functional.interpolate(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=False))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            depthwise_separable=True,
            **kwargs)
    elif type == 'conv':
        return ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


class DAFormerHead(BaseHead):

    def __init__(self,
                 in_channels: List[int],
                 in_index: Union[List[int], int],
                 num_classes: int,
                 input_transform: Optional[str] = None,
                 channels: int = 256,
                 dropout_ratio: float = 0.1,
                 embed_dims: int = 256,
                 ):
        super().__init__(num_classes, in_index, input_transform)
        self.in_channels = in_channels
        self.channels = channels
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_channels)

        self.embed_layers = {}
        for i, (in_channels, embed_dim) in enumerate(zip(self.in_channels, embed_dims)):
            if i == len(self.in_channels) - 1:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, type='mlp')
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, type='mlp')
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, type='aspp', sep=True, dilations=(1, 6, 12, 18), pool=False, activation_layer=nn.ReLU, norm_layer=nn.BatchNorm2d)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)
        # mmseg init
        nn.init.normal_(self.conv_seg.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv_seg.bias, 0)
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

    def forward(self, x):
        x = self._transform_inputs(x)
        n, _, h, w = x[-1].shape

        os_size = x[0].size()[2:]
        _c_list = []
        for i in range(len(self.in_channels)):
            _c = self.embed_layers[str(i)](x[i])
            if _c.dim() == 3:
                _c = _c.permute(0, 2, 1).contiguous().reshape(
                    n, -1, x[i].shape[2], x[i].shape[3])
            if _c.size()[2:] != os_size:
                _c = nn.functional.interpolate(
                    _c,
                    size=os_size,
                    mode='bilinear',
                    align_corners=False)
            _c_list.append(_c)

        x = self.fuse_layer(torch.cat(_c_list, dim=1))

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv_seg(x)
        return x
