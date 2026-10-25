"""DMVFN inference architecture adapted for deterministic, device-safe inference.

Derived from https://github.com/hzwer/CVPR2023-DMVFN (CVPR 2023).
Copyright 2023 Megvii Inc. MIT License; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_grid_cache: dict[tuple, Tensor] = {}


def warp(image: Tensor, flow: Tensor) -> Tensor:
    key = (image.device.type, image.device.index, image.dtype, tuple(image.shape))
    if key not in _grid_cache:
        horizontal = torch.linspace(-1.0, 1.0, image.shape[3], device=image.device, dtype=image.dtype)
        vertical = torch.linspace(-1.0, 1.0, image.shape[2], device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(vertical, horizontal, indexing="ij")
        _grid_cache[key] = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(image.shape[0], -1, -1, -1)
        if len(_grid_cache) > 8:
            del _grid_cache[next(iter(_grid_cache))]
    normalized = torch.stack(
        (flow[:, 0] / ((image.shape[3] - 1.0) / 2.0), flow[:, 1] / ((image.shape[2] - 1.0) / 2.0)),
        dim=-1,
    )
    return F.grid_sample(
        image, _grid_cache[key] + normalized, mode="bilinear", padding_mode="border", align_corners=True
    )


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return torch.bernoulli(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def deconv(in_planes: int, out_planes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes),
    )


def conv(in_planes: int, out_planes: int, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


class MVFB(nn.Module):
    def __init__(self, in_planes: int, num_feature: int):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, num_feature // 2, 3, 2, 1),
            conv(num_feature // 2, num_feature, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
        )
        self.conv_sq = conv(num_feature, num_feature // 4)
        self.conv1 = nn.Sequential(conv(in_planes, 8, 3, 2, 1))
        self.convblock1 = nn.Sequential(conv(8, 8))
        self.lastconv = nn.ConvTranspose2d(num_feature // 4 + 8, 5, 4, 2, 1)

    def forward(self, x: Tensor, flow: Tensor, scale: int):
        x0, flow0 = x, flow
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
        x = torch.cat((x, flow), 1)
        x1 = self.conv0(x)
        x2 = self.conv_sq(self.convblock(x1) + x1)
        x2 = F.interpolate(x2, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        x3 = self.conv1(torch.cat((x0, flow0), 1))
        x4 = self.convblock1(x3)
        result = self.lastconv(torch.cat((x2, x4), dim=1))
        return result[:, :4], result[:, 4:5]


class DMVFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = MVFB(17, num_feature=160)
        self.block1 = MVFB(17, num_feature=160)
        self.block2 = MVFB(17, num_feature=160)
        self.block3 = MVFB(17, num_feature=80)
        self.block4 = MVFB(17, num_feature=80)
        self.block5 = MVFB(17, num_feature=80)
        self.block6 = MVFB(17, num_feature=44)
        self.block7 = MVFB(17, num_feature=44)
        self.block8 = MVFB(17, num_feature=44)
        self.routing = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.l1 = nn.Linear(32, 9)

    def forward(self, x: Tensor, scale=(4, 4, 4, 2, 2, 2, 1, 1, 1), training=True, deterministic=True):
        batch_size, _, height, width = x.shape
        if not training and batch_size != 1:
            raise ValueError("DMVFN inference currently requires batch_size=1")
        routing_vector = torch.sigmoid(self.l1(self.routing(x[:, :6]).reshape(batch_size, -1)))
        routing_vector = routing_vector / (routing_vector.sum(1, True) + 1e-6) * 4.5
        routing_vector = torch.clamp(routing_vector, 0, 1)
        if training or not deterministic:
            ref = RoundSTE.apply(routing_vector)
        else:
            ref = (routing_vector >= 0.5).to(routing_vector.dtype)

        img0, img1 = x[:, :3], x[:, 3:6]
        flow_list, merged_final, mask_final = [], [], []
        warped_img0, warped_img1 = img0, img1
        flow = x.new_zeros(batch_size, 4, height, width)
        mask = x.new_zeros(batch_size, 1, height, width)
        blocks = [getattr(self, f"block{index}") for index in range(9)]

        if training:
            for index, block in enumerate(blocks):
                flow_d, mask_d = block(
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale[index]
                )
                flow_now, mask_now = flow + flow_d, mask + mask_d
                flow = flow + flow_d * ref[:, index].reshape(batch_size, 1, 1, 1)
                mask = mask + mask_d * ref[:, index].reshape(batch_size, 1, 1, 1)
                flow_list.append(flow)
                warped_now0, warped_now1 = warp(img0, flow_now[:, :2]), warp(img1, flow_now[:, 2:4])
                if index < 8:
                    mask_final.append(torch.sigmoid(mask_now))
                    merged_final.append((warped_now0, warped_now1))
                else:
                    mask_final.append(torch.sigmoid(mask))
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    merged_final.append((warped_img0, warped_img1))
                warped_img0, warped_img1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        else:
            for index, block in enumerate(blocks):
                if bool(ref[0, index]):
                    flow_d, mask_d = block(
                        torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale[index]
                    )
                    flow, mask = flow + flow_d, mask + mask_d
                    mask_final.append(torch.sigmoid(mask))
                    flow_list.append(flow)
                    warped_img0, warped_img1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
                    merged_final.append((warped_img0, warped_img1))
        for index, (first, second) in enumerate(merged_final):
            merged_final[index] = (first * mask_final[index] + second * (1 - mask_final[index])).clamp(0, 1)
        return merged_final
