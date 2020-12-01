from typing import Dict

import numpy as np
import torch
from torch import nn as nn

from habitat_baselines.utils.common import Flatten


class DeepVOCNN(nn.Module):
    """An implementation of the DeepVO model 

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(
        self,
        observation_space,
        output_size,
    ):
        super().__init__()

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )
##########################Start DeepVO Definition###############################
            self.cnn =nn.Sequential(
            nn.AvgPool2d(5,1,5//2),
            #conv in based on # of input channels
            nn.Conv2d(4, 32, kernel_size=7, stride=2, padding=(7-1)//2, bias=True),
            nn.ReLU(True),
            nn.Dropout(.1),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=(5-1)//2, bias=True),
            nn.ReLU(True),
            nn.Dropout(.1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=(5-1)//2, bias=True),
            nn.ReLU(True),
            nn.Dropout(.1),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=(4-1)//2, bias=True),
            nn.ReLU(True),
            nn.Dropout(.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(3-1)//2, bias=True),
            nn.ReLU(True),
            nn.Dropout(.1),
            )
            self.rnn = nn.LSTM(input_size=220,hidden_size=512,num_layers=2,batch_first=True)
            self.rnn_drop_out = nn.Dropout(0.2)
            self.flatten = Flatten()
            self.linear = nn.Linear(65536, 512)
######################## End Deep VO Definition ###############################

        self.layer_init()

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
            elif isinstance(layer, nn.LSTM):
                # layer 1
                kaiming_normal_(layer.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(layer.weight_hh_l0)
                layer.bias_ih_l0.data.zero_()
                layer.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = layer.bias_hh_l0.size(0)
                start, end = n//4, n//2
                layer.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(layer.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
                kaiming_normal_(layer.weight_hh_l1)
                layer.bias_ih_l1.data.zero_()
                layer.bias_hh_l1.data.zero_()
                n = layer.bias_hh_l1.size(0)
                start, end = n//4, n//2
                layer.bias_hh_l1.data[start:end].fill_(1.)
    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0
    def encode_image(self, x):
        out = self.cnn(x)
        return out
    def forward(self, observations: Dict[str, torch.Tensor]):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        seq_len = cnn_input.size(1)
        batch = cnn_input.size(0)
        # CNN
        x = self.encode_image(cnn_input)
        x = x.view(batch,x.size(1),-1)
        # RNN
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out 
