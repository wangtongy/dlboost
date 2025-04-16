__all__ = ["DWUNet", "DWUNet_P2PCSE"]
from math import prod

# from torch.nn import functional as F
from typing import Optional, Sequence

import torch

# from monai.utils import ensure_tuple_rep
from einops.layers.torch import Rearrange
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# from dlboost.models.BasicUNet import Down


class DWInvertedBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels,
        out_channels,
        #  drop_path=0.,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str,
        dropout: tuple | str | float | None = None,
        layer_scale_init_value=1e-6,
        mlp_ratio=4.0,
        padding=5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if spatial_dims == 2:
            conv = nn.Conv2d
        elif spatial_dims == 3:
            conv = nn.Conv3d
        if in_channels != out_channels:
            self.shortcut_conv = conv(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=True
            )
        self.pwconv1 = conv(
            in_channels, int(mlp_ratio * out_channels), kernel_size=1, bias=True
        )  # pointwise/1x1 convs, implemented with linear layers
        # self.norm1 = norm_layer(int(mlp_ratio * dim))
        self.norm1 = get_norm_layer(
            name=norm_name,
            spatial_dims=spatial_dims,
            channels=int(mlp_ratio * out_channels),
        )
        self.act1 = get_act_layer(act_name)
        self.conv_dw = conv(
            int(mlp_ratio * out_channels),
            int(mlp_ratio * out_channels),
            kernel_size=kernel_size,
            padding="same",
            groups=int(mlp_ratio * out_channels),
            stride=stride,
            bias=True,
        )  # depthwise conv
        self.pwconv2 = conv(
            int(mlp_ratio * out_channels), out_channels, kernel_size=1, bias=True
        )
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.downsample = downsample

    def forward(self, x):
        shortcut = x

        x = self.pwconv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.pwconv2(x)
        if self.in_channels != self.out_channels:
            shortcut = self.shortcut_conv(shortcut)
        x = x + shortcut
        return x


class DWInvertedDownStage(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels,
        out_channels,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str,
        dropout: tuple | str | float | None = None,
        blocks_num: int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        if spatial_dims == 2:
            self.downsample = nn.AvgPool2d(stride)
        elif spatial_dims == 3:
            self.downsample = nn.AvgPool3d(stride)
        blocks = [self.downsample]
        blocks.append(
            DWInvertedBlock(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size,
                1,
                norm_name,
                act_name,
                dropout,
            )
        )
        for _ in range(blocks_num - 1):
            self.blocks.append(
                DWInvertedBlock(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size,
                    1,
                    norm_name,
                    act_name,
                    dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class DWInvertedUpStage(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels,
        cat_channels: int,
        out_channels,
        upsample_factors,
        kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str,
        dropout: tuple | str | float | None = None,
        blocks_num: int = 2,
    ):
        super().__init__()
        if spatial_dims == 2:
            self.upsample = nn.Upsample(
                scale_factor=upsample_factors, mode="bilinear", align_corners=True
            )
        elif spatial_dims == 3:
            self.upsample = nn.Upsample(
                scale_factor=upsample_factors, mode="trilinear", align_corners=True
            )
        else:
            raise NotImplementedError("Only 2D and 3D are supported.")
        blocks = [
            DWInvertedBlock(
                spatial_dims,
                in_channels + cat_channels,
                out_channels,
                kernel_size,
                1,
                norm_name,
                act_name,
                dropout,
            )
        ]
        for _ in range(blocks_num - 1):
            blocks.append(
                DWInvertedBlock(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size,
                    1,
                    norm_name,
                    act_name,
                    dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        # self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)
        x = self.blocks(
            torch.cat([x_e, x_0], dim=1)
        )  # input channels: (cat_chns + up_chns)
        return x


class DWInvertedPatchExpand(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cat_channels: int,
        expand_ratio,
    ):
        super().__init__()
        self.expand = nn.Conv3d(
            in_channels, out_channels * prod(expand_ratio), kernel_size=1
        )
        self.shuffle = Rearrange(
            "b (c p1 p2 p3) d h w-> b c (d p1) (h p2) (w p3)",
            p1=expand_ratio[0],
            p2=expand_ratio[1],
            p3=expand_ratio[2],
        )
        self.contract = nn.Conv3d(
            out_channels + cat_channels, out_channels, kernel_size=1
        )
        # self.expand = nn.Conv3d(in_channels, out_channels * prod(expand_ratio), kernel_size=1)
        # self.shuffle = Rearrange('b (c p1 p2 p3) d h w-> b c (d p1) (h p2) (w p3)', p1=expand_ratio[0], p2=expand_ratio[1], p3=expand_ratio[2])

    def forward(self, x, x0):
        x = self.expand(x)
        x = self.shuffle(x)
        x = self.contract(torch.cat([x, x0], dim=1))
        return x


class DWUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
        kernel_sizes=((1, 7, 7), (1, 7, 7), (1, 7, 7), (1, 7, 7), (1, 7, 7)),
        features: Sequence[int] = (32, 64, 128, 256, 512),
        # stages = (2,2,2,2),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        dropout: float | tuple = 0.0,
    ):
        super().__init__()
        # fea = ensure_tuple_rep(features, 6)
        # print(f"BasicUNet features: {fea}.")
        # features = [in_channels*prod(stem_patch_size)*(2**i) for i in range(5)]
        # self.stem = Rearrange('b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1=stem_patch_size[0], p2=stem_patch_size[1], p3=stem_patch_size[2])

        self.conv_0 = nn.Sequential(
            *[
                DWInvertedBlock(
                    spatial_dims, in_channels, features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
            ]
        )

        self.down_1 = DWInvertedDownStage(
            spatial_dims,
            features[0],
            features[1],
            kernel_sizes[0],
            strides[0],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.down_2 = DWInvertedDownStage(
            spatial_dims,
            features[1],
            features[2],
            kernel_sizes[1],
            strides[1],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.down_3 = DWInvertedDownStage(
            spatial_dims,
            features[2],
            features[3],
            kernel_sizes[2],
            strides[2],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.down_4 = DWInvertedDownStage(
            spatial_dims,
            features[3],
            features[4],
            kernel_sizes[3],
            strides[3],
            norm,
            act,
            dropout,
            blocks_num=2,
        )

        self.upcat_4 = DWInvertedUpStage(
            spatial_dims,
            features[4],
            features[3],
            features[3],
            strides[3],
            kernel_sizes[3],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.upcat_3 = DWInvertedUpStage(
            spatial_dims,
            features[3],
            features[2],
            features[2],
            strides[2],
            kernel_sizes[2],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.upcat_2 = DWInvertedUpStage(
            spatial_dims,
            features[2],
            features[1],
            features[1],
            strides[1],
            kernel_sizes[1],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.upcat_1 = DWInvertedUpStage(
            spatial_dims,
            features[1],
            features[0],
            features[0],
            strides[0],
            kernel_sizes[0],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.final_conv = nn.Sequential(
            *[
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], out_channels, 3, 1, norm, act, dropout
                ),
            ]
        )
        # self.final_conv = nn.Sequential(*[DWInvertedBlock(spatial_dims, features[0], features[0], 3, 1, norm, act, dropout) for i in range(2)])
        # self.expand = nn.ConvTranspose3d(features[0], in_channels, kernel_size=stem_patch_size, stride=stem_patch_size)
        # self.expand = DWInvertedPatchExpand(features[0], in_channels, in_channels, stem_patch_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        # x0 = self.stem(x)

        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        # logits = self.expand(u1, x)

        return logits


class DWUNet_reduced(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
        kernel_sizes=((1, 7, 7), (1, 7, 7), (1, 7, 7), (1, 7, 7)),
        features: Sequence[int] = (32, 64, 128, 256),
        # stages = (2,2,2,2),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        dropout: float | tuple = 0.0,
    ):
        super().__init__()

        self.conv_0 = nn.Sequential(
            *[
                DWInvertedBlock(
                    spatial_dims, in_channels, features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
            ]
        )

        self.down_1 = DWInvertedDownStage(
            spatial_dims,
            features[0],
            features[1],
            kernel_sizes[0],
            strides[0],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.down_2 = DWInvertedDownStage(
            spatial_dims,
            features[1],
            features[2],
            kernel_sizes[1],
            strides[1],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.down_3 = DWInvertedDownStage(
            spatial_dims,
            features[2],
            features[3],
            kernel_sizes[2],
            strides[2],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
    
        # self.down_4 = DWInvertedDownStage(
        #     spatial_dims,
        #     features[3],
        #     features[4],
        #     kernel_sizes[3],
        #     strides[3],
        #     norm,
        #     act,
        #     dropout,
        #     blocks_num=2,
        # )

        # self.upcat_4 = DWInvertedUpStage(
        #     spatial_dims,
        #     features[4],
        #     features[3],
        #     features[3],
        #     strides[3],
        #     kernel_sizes[3],
        #     norm,
        #     act,
        #     dropout,
        #     blocks_num=2,
        # )
        self.upcat_3 = DWInvertedUpStage(
            spatial_dims,
            features[3],
            features[2],
            features[2],
            strides[2],
            kernel_sizes[2],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.upcat_2 = DWInvertedUpStage(
            spatial_dims,
            features[2],
            features[1],
            features[1],
            strides[1],
            kernel_sizes[1],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.upcat_1 = DWInvertedUpStage(
            spatial_dims,
            features[1],
            features[0],
            features[0],
            strides[0],
            kernel_sizes[0],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.final_conv = nn.Sequential(
            *[
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], out_channels, 3, 1, norm, act, dropout
                ),

            ]
        )
        # self.final_conv = nn.Sequential(*[DWInvertedBlock(spatial_dims, features[0], features[0], 3, 1, norm, act, dropout) for i in range(2)])
        # self.expand = nn.ConvTranspose3d(features[0], in_channels, kernel_size=stem_patch_size, stride=stem_patch_size)
        # self.expand = DWInvertedPatchExpand(features[0], in_channels, in_channels, stem_patch_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        # x0 = self.stem(x)

        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        # x4 = self.down_4(x3)

        # u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(x3, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)

        return logits

class DWUNet_Checkpointing(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        strides=((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
        kernel_sizes=((1, 7, 7), (1, 7, 7), (1, 7, 7), (1, 7, 7), (1, 7, 7)),
        features: Sequence[int] = (32, 64, 128, 256, 512),
        # stages = (2,2,2,2),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        dropout: float | tuple = 0.0,
    ):
        super().__init__()
        # fea = ensure_tuple_rep(features, 6)
        # print(f"BasicUNet features: {fea}.")
        # features = [in_channels*prod(stem_patch_size)*(2**i) for i in range(5)]
        # self.stem = Rearrange('b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1=stem_patch_size[0], p2=stem_patch_size[1], p3=stem_patch_size[2])

        self.conv_0 = nn.Sequential(
            *[
                DWInvertedBlock(
                    spatial_dims, in_channels, features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
            ]
        )

        self.down_1 = DWInvertedDownStage(
            spatial_dims,
            features[0],
            features[1],
            kernel_sizes[0],
            strides[0],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.down_2 = DWInvertedDownStage(
            spatial_dims,
            features[1],
            features[2],
            kernel_sizes[1],
            strides[1],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.down_3 = DWInvertedDownStage(
            spatial_dims,
            features[2],
            features[3],
            kernel_sizes[2],
            strides[2],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.down_4 = DWInvertedDownStage(
            spatial_dims,
            features[3],
            features[4],
            kernel_sizes[3],
            strides[3],
            norm,
            act,
            dropout,
            blocks_num=2,
        )

        self.upcat_4 = DWInvertedUpStage(
            spatial_dims,
            features[4],
            features[3],
            features[3],
            strides[3],
            kernel_sizes[3],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.upcat_3 = DWInvertedUpStage(
            spatial_dims,
            features[3],
            features[2],
            features[2],
            strides[2],
            kernel_sizes[2],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.upcat_2 = DWInvertedUpStage(
            spatial_dims,
            features[2],
            features[1],
            features[1],
            strides[1],
            kernel_sizes[1],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.upcat_1 = DWInvertedUpStage(
            spatial_dims,
            features[1],
            features[0],
            features[0],
            strides[0],
            kernel_sizes[0],
            norm,
            act,
            dropout,
            blocks_num=2,
        )
        self.final_conv = nn.Sequential(
            *[
                DWInvertedBlock(
                    spatial_dims, features[0], features[0], 3, 1, norm, act, dropout
                ),
                DWInvertedBlock(
                    spatial_dims, features[0], out_channels, 3, 1, norm, act, dropout
                ),
            ]
        )
        # self.final_conv = nn.Sequential(*[DWInvertedBlock(spatial_dims, features[0], features[0], 3, 1, norm, act, dropout) for i in range(2)])
        # self.expand = nn.ConvTranspose3d(features[0], in_channels, kernel_size=stem_patch_size, stride=stem_patch_size)
        # self.expand = DWInvertedPatchExpand(features[0], in_channels, in_channels, stem_patch_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        x0 = checkpoint_sequential(self.conv_0, 2, x)
        x1 = checkpoint(self.down_1, x0)
        x2 = checkpoint(self.down_2, x1)
        x3 = checkpoint(self.down_3, x2)
        x4 = checkpoint(self.down_4, x3)
        # x1 = self.down_1(x0)
        # x2 = self.down_2(x1)
        # x3 = self.down_3(x2)
        # x4 = self.down_4(x3)

        u4 = checkpoint(self.upcat_4, x4, x3)
        u3 = checkpoint(self.upcat_3, u4, x2)
        u2 = checkpoint(self.upcat_2, u3, x1)
        u1 = checkpoint(self.upcat_1, u2, x0)

        # u4 = self.upcat_4(x4, x3)
        # u3 = self.upcat_3(u4, x2)
        # u2 = self.upcat_2(u3, x1)
        # u1 = self.upcat_1(u2, x0)

        logits = checkpoint_sequential(self.final_conv, 2, u1)
        return logits
