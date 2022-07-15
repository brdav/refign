# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/open-mmlab/mmclassification/tree/master/mmcls/models/backbones
import os
from typing import Optional

import torch
import torch.nn as nn

from ..modules import BasicBlock, Bottleneck

# imagenet pretrained
model_urls = {
    "resnet18_v1c": "https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth",
    "resnet50_v1c": "https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth",
    "resnet101_v1c": "https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth",
}


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.
    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 norm_layer=nn.BatchNorm2d,
                 contract_dilation=False,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                norm_layer(out_channels * block.expansion)
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if dilation > 1 and contract_dilation:
            first_dilation = dilation // 2
        else:
            first_dilation = dilation
        layers.append(
            block(
                inplanes=in_channels,
                planes=out_channels,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                norm_layer=norm_layer,
                **kwargs))
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=in_channels,
                    planes=out_channels,
                    stride=1,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class ResNet(nn.Module):
    """ResNet backbone.
    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.
    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    Example:
        >>> from mmcls.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        'resnet18_v1c': {
            'depth': 18,
            'block': BasicBlock,
            'stage_blocks': (2, 2, 2, 2),
            'deep_stem': True,
            'avg_down': False,
            'cifar_version': False
        },
        'resnet50_v1c': {
            'depth': 50,
            'block': Bottleneck,
            'stage_blocks': (3, 4, 6, 3),
            'deep_stem': True,
            'avg_down': False,
            'cifar_version': False
        },
        'resnet101_v1c': {
            'depth': 101,
            'block': Bottleneck,
            'stage_blocks': (3, 4, 23, 3),
            'deep_stem': True,
            'avg_down': False,
            'cifar_version': False
        },
    }

    def __init__(self,
                 model_type: str,
                 pretrained: Optional[str] = None,
                 channels_last: Optional[int] = None,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 contract_dilation=False,
                 max_pool_ceil_mode=False):
        norm_layer = nn.BatchNorm2d
        super(ResNet, self).__init__()
        self.model_type = model_type
        self.depth = self.arch_settings[model_type]['depth']
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = self.arch_settings[model_type]['deep_stem']
        self.avg_down = self.arch_settings[model_type]['avg_down']
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block = self.arch_settings[model_type]['block']
        stage_blocks = self.arch_settings[model_type]['stage_blocks']
        self.stage_blocks = stage_blocks[:num_stages]
        self.contract_dilation = contract_dilation
        self.max_pool_ceil_mode = max_pool_ceil_mode

        self._make_stem_layer(in_channels, stem_channels, norm_layer)

        self.res_layers = []
        _in_channels = stem_channels
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            _out_channels = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                norm_layer=norm_layer,
                contract_dilation=contract_dilation)
            _in_channels = _out_channels * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        self.init_weights(pretrained=pretrained)

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def _make_stem_layer(self, in_channels, stem_channels, norm_layer):
        if self.deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                norm_layer(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                norm_layer(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                norm_layer(stem_channels),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1 = norm_layer(stem_channels)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=self.max_pool_ceil_mode)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, (Bottleneck)):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, (BasicBlock)):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

        if pretrained is not None:
            if pretrained == 'imagenet':
                pretrained = model_urls[self.model_type]

            if os.path.exists(pretrained):
                checkpoint = torch.load(pretrained)
            elif os.path.exists(os.path.join(os.environ.get('TORCH_HOME', ''), 'hub', pretrained)):
                checkpoint = torch.load(os.path.join(
                    os.environ.get('TORCH_HOME', ''), 'hub', pretrained))
            else:
                checkpoint = torch.hub.load_state_dict_from_url(
                    pretrained, progress=True)
            if 'state_dict' in checkpoint.keys():
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith('fc.')}
            self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
