"""Pinned TransNetV2 PyTorch inference model.

Source: https://github.com/soCzech/TransNetV2/blob/master/inference-pytorch/transnetv2_pytorch.py
Retrieved: 2026-08-04. The upstream MIT license is included beside this file.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as functional


class TransNetV2(nn.Module):
    def __init__(
        self,
        F=16,
        L=3,
        S=2,
        D=1024,
        use_many_hot_targets=True,
        use_frame_similarity=True,
        use_color_histograms=True,
        use_mean_pooling=False,
        dropout_rate=0.5,
        use_convex_comb_reg=False,
        use_resnet_features=False,
        use_resnet_like_top=False,
        frame_similarity_on_last_layer=False,
    ):
        super().__init__()
        unsupported = (
            use_resnet_features
            or use_resnet_like_top
            or use_convex_comb_reg
            or frame_similarity_on_last_layer
        )
        if unsupported:
            raise NotImplementedError("Some options are not implemented in the PyTorch model")
        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0)]
            + [
                StackedDDCNNV2(
                    in_filters=(F * 2 ** (i - 1)) * 4,
                    n_blocks=S,
                    filters=F * 2**i,
                )
                for i in range(1, L)
            ]
        )
        self.frame_sim_layer = (
            FrameSimilarity(
                sum((F * 2**i) * 4 for i in range(L)),
                lookup_window=101,
                output_dim=128,
                similarity_dim=128,
                use_bias=True,
            )
            if use_frame_similarity
            else None
        )
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None
        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128
        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None
        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        assert list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8
        x = inputs.permute([0, 4, 1, 2, 3]).float().div_(255.0)
        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)
        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4]).permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)
        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)
        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)
        x = functional.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        one_hot = self.cls_layer1(x)
        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}
        return one_hot


class StackedDDCNNV2(nn.Module):
    def __init__(
        self,
        in_filters,
        n_blocks,
        filters,
        shortcut=True,
        use_octave_conv=False,
        pool_type="avg",
        stochastic_depth_drop_prob=0.0,
    ):
        super().__init__()
        if use_octave_conv:
            raise NotImplementedError("Octave convolution is not implemented")
        assert pool_type in {"max", "avg"}
        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    activation=functional.relu if i != n_blocks else None,
                )
                for i in range(1, n_blocks + 1)
            ]
        )
        pool = nn.MaxPool3d if pool_type == "max" else nn.AvgPool3d
        self.pool = pool(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None
        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x
        x = functional.relu(x)
        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob and self.training:
                x = shortcut if random.random() < self.stochastic_depth_drop_prob else x + shortcut
            elif self.stochastic_depth_drop_prob:
                x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut
        return self.pool(x)


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, batch_norm=True, activation=None, octave_conv=False):
        super().__init__()
        if octave_conv:
            raise NotImplementedError("Octave convolution is not implemented")
        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)
        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        x = torch.cat(
            [self.Conv3D_1(inputs), self.Conv3D_2(inputs), self.Conv3D_4(inputs), self.Conv3D_8(inputs)],
            dim=1,
        )
        if self.bn is not None:
            x = self.bn(x)
        return self.activation(x) if self.activation is not None else x


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_filters, filters, dilation_rate, separable=True, octave=False, use_bias=True, kernel_initializer=None):
        super().__init__()
        if octave or kernel_initializer is not None:
            raise NotImplementedError("Requested convolution option is not implemented")
        if separable:
            self.layers = nn.ModuleList(
                [
                    nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                    nn.Conv3d(
                        2 * filters,
                        filters,
                        kernel_size=(3, 1, 1),
                        dilation=(dilation_rate, 1, 1),
                        padding=(dilation_rate, 0, 0),
                        bias=use_bias,
                    ),
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    nn.Conv3d(
                        in_filters,
                        filters,
                        kernel_size=3,
                        dilation=(dilation_rate, 1, 1),
                        padding=(dilation_rate, 1, 1),
                        bias=use_bias,
                    )
                ]
            )

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(self, in_filters, similarity_dim=128, lookup_window=101, output_dim=128, stop_gradient=False, use_bias=False):
        super().__init__()
        if stop_gradient:
            raise NotImplementedError("stop_gradient is not implemented")
        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1

    def forward(self, inputs):
        x = torch.cat([torch.mean(value, dim=[3, 4]) for value in inputs], dim=1).transpose(1, 2)
        x = functional.normalize(self.projection(x), p=2, dim=2)
        batch_size, time_window = x.shape[:2]
        similarities = torch.bmm(x, x.transpose(1, 2))
        similarities_padded = functional.pad(
            similarities,
            [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2],
        )
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).repeat(1, time_window, self.lookup_window)
        time_indices = torch.arange(time_window, device=x.device).view(1, -1, 1).repeat(batch_size, 1, self.lookup_window)
        lookup_indices = torch.arange(self.lookup_window, device=x.device).view(1, 1, -1).repeat(batch_size, time_window, 1)
        lookup_indices = lookup_indices + time_indices
        return functional.relu(self.fc(similarities_padded[batch_indices, time_indices, lookup_indices]))


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super().__init__()
        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()
        batch_size, time_window, height, width, channels = frames.shape
        assert channels == 3
        flat = frames.view(batch_size * time_window, height * width, 3)
        red, green, blue = flat[:, :, 0] >> 5, flat[:, :, 1] >> 5, flat[:, :, 2] >> 5
        bins = (red << 6) + (green << 3) + blue
        prefixes = (torch.arange(batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        bins = (bins + prefixes).view(-1)
        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, bins, torch.ones(len(bins), dtype=torch.int32, device=frames.device))
        histograms = histograms.view(batch_size, time_window, 512).float()
        return functional.normalize(histograms, p=2, dim=2)

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)
        batch_size, time_window = x.shape[:2]
        similarities = torch.bmm(x, x.transpose(1, 2))
        padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).repeat(1, time_window, self.lookup_window)
        time_indices = torch.arange(time_window, device=x.device).view(1, -1, 1).repeat(batch_size, 1, self.lookup_window)
        lookup_indices = torch.arange(self.lookup_window, device=x.device).view(1, 1, -1).repeat(batch_size, time_window, 1)
        lookup_indices = lookup_indices + time_indices
        result = padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(result)) if self.fc is not None else result
