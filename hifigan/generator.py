# adapted from https://github.com/jik876/hifi-gan/blob/master/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from typing import Optional, Tuple

from hifigan.utils import get_padding
from hifigan.constants import UNIT_MODE, MEL_SPECTROGRAM_MODE, mel_bins
LRELU_SLOPE = 0.1


class HifiganGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = mel_bins,
        unit_nums: Optional[int] = None,
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        upsample_kernel_sizes: Tuple[int, ...] = (20, 8, 4, 4),
        upsample_initial_channel: int = 512,
        upsample_factors: int = (10, 4, 2, 2),
        inference_padding: int = 5,
        sample_rate: int = 16000,
    ):
        r"""HiFiGAN Generator
        Args:
            in_channels (int): number of input channels.
            unit_nums (Optional[int]): number of HuBERT label classes when HuBERT label is used as input.
            resblock_dilation_sizes (Tuple[Tuple[int, ...], ...]): list of dilation values in each layer of a `ResBlock`.
            resblock_kernel_sizes (Tuple[int, ...]): list of kernel sizes for each `ResBlock`.
            upsample_kernel_sizes (Tuple[int, ...]): list of kernel sizes for each transposed convolution.
            upsample_initial_channel (int): number of channels for the first upsampling layer. This is divided by 2
                for each consecutive upsampling layer.
            upsample_factors (Tuple[int, ...]): upsampling factors (stride) for each upsampling layer.
            inference_padding (int): constant padding applied to the input at inference time.
            sample_rate (int): sample rate of the generated audio.
        """
        super().__init__()
        self.mode = UNIT_MODE if unit_nums is not None else MEL_SPECTROGRAM_MODE
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.sample_rate = sample_rate

        # initial upsampling layers
        if self.mode == UNIT_MODE:
            # lookup table as in https://arxiv.org/abs/2104.00355
            # The extra embedding to the end is used for padding in dataset collating.
            self.lut = nn.Embedding(unit_nums+1, in_channels)
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        )

        # upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock(ch, k, d))

        # post convolution layer
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): mel spectrogram or unit sequence

        Returns:
            torch.Tensor: generated waveform
        """
        if self.mode == UNIT_MODE:
            x = self.lut(x).transpose(-1, -2)
        o = self.conv_pre(x)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)

        for layer in self.resblocks:
            layer.remove_weight_norm()

        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class ResBlock(torch.nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3, 5)
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
