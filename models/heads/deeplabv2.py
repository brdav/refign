from typing import List, Optional, Union

import torch.nn as nn

from .base import BaseHead


class DeepLabV2Head(BaseHead):
    def __init__(self,
                 in_channels: int,
                 in_index: Union[List[int], int],
                 num_classes: int,
                 input_transform: Optional[str] = None,
                 dilation_series: List[int] = [6, 12, 18, 24],
                 padding_series: List[int] = [6, 12, 18, 24]):
        super().__init__(num_classes, in_index, input_transform)
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        x = self._transform_inputs(x)
        return sum(stage(x) for stage in self.conv2d_list)
