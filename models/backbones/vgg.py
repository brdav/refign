import os
from typing import Dict, List, Optional, Union, cast

import torch
import torch.nn as nn

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):

    arch_settings = {
        'vgg11': {
            'cfg': 'A',
            'batch_norm': False,
        },
        'vgg11_bn': {
            'cfg': 'A',
            'batch_norm': True,
        },
        'vgg13': {
            'cfg': 'B',
            'batch_norm': False,
        },
        'vgg13_bn': {
            'cfg': 'B',
            'batch_norm': True,
        },
        'vgg16': {
            'cfg': 'D',
            'batch_norm': False,
        },
        'vgg16_bn': {
            'cfg': 'D',
            'batch_norm': True,
        },
        'vgg19': {
            'cfg': 'E',
            'batch_norm': False,
        },
        'vgg19_bn': {
            'cfg': 'E',
            'batch_norm': True,
        }
    }

    def __init__(
        self, model_type: str, out_indices: list = [0, 1, 2, 3, 4, 5], pretrained: Optional[str] = None
    ) -> None:
        super().__init__()
        self.model_type = model_type
        cfg = self.arch_settings[model_type]['cfg']
        batch_norm = self.arch_settings[model_type]['batch_norm']
        self.features, layer_indices = self._make_layers(
            cfgs[cfg], batch_norm=batch_norm)
        self.layer_indices = [layer_indices[i] for i in out_indices]
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrained is not None:
            if pretrained == 'imagenet':
                pretrained = model_urls[self.model_type]

            if os.path.exists(pretrained):
                checkpoint = torch.load(pretrained)
            else:
                checkpoint = torch.hub.load_state_dict_from_url(
                    pretrained, progress=True)
            if 'state_dict' in checkpoint.keys():
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict = {k: v for k, v in state_dict.items(
            ) if not k.startswith('classifier.')}
            self.load_state_dict(state_dict, strict=True)

    def forward(self, x: torch.Tensor, extract_only_indices=None) -> torch.Tensor:
        if extract_only_indices:
            layer_indices = [self.layer_indices[i]
                             for i in extract_only_indices]
        else:
            layer_indices = self.layer_indices
        prev_i = 0
        outs = []
        for i in layer_indices:
            x = self.features[prev_i:i](x)
            outs.append(x)
            prev_i = i
        return outs

    @staticmethod
    def _make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 3

        current_idx = 0
        layer_indices = []
        first_relu = True
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                current_idx += 1
                layer_indices.append(current_idx)
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    current_idx += 3
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    current_idx += 2
                in_channels = v
                if first_relu:
                    first_relu = False
                    layer_indices.append(current_idx)
        return nn.Sequential(*layers), layer_indices
