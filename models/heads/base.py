from typing import Iterable, List, Optional, Union

import torch
import torch.nn as nn


class BaseHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_index: Union[List[int], int],
                 input_transform: Optional[str] = None):
        super().__init__()
        self.input_transform = input_transform
        if isinstance(in_index, Iterable) and len(in_index) == 1:
            self.in_index = in_index[0]
        else:
            self.in_index = in_index
        self.num_classes = num_classes

    def forward(self, inp):
        raise NotImplementedError

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                nn.functional.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs
