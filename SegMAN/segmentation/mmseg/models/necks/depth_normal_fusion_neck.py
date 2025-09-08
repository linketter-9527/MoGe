# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
# import cv2
# import numpy as np
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import NECKS


class _ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, 1, 1)
        u = F.adaptive_avg_pool2d(x, 1)
        u = self.fc2(self.act(self.fc1(u)))
        return torch.sigmoid(u)


@NECKS.register_module()
class DepthNormalFusionNeck(nn.Module):
    """
    Depth+Normal Guided Fusion Neck for SegMAN.

    This neck fuses RGB backbone multi-scale features with geometric cues
    (1-channel depth and 3-channel normal maps) produced by an auxiliary
    backbone, and outputs refined multi-scale features with the SAME channel
    dimensions as the RGB backbone to keep the decode head unchanged.

    Acceptable input formats (passed from segmentor.extract_feat -> neck):
      - List/Tuple of length 6:
        [x1, x2, x3, x4, depth, normal]
        where x{i} are RGB features at 1/4,1/8,1/16,1/32 resolutions respectively,
        depth is (B,1,Hd,Wd), normal is (B,3,Hn,Wn).
      - Dict with keys {'rgb_feats': [x1..x4], 'depth': d, 'normal': n}
      - List/Tuple of length 5:
        [x1, x2, x3, x4, dn] where dn is concatenated depth+normal (B,4,H,W)

    Args:
        in_channels (list[int]): Channels for the 4 RGB feature levels.
        dn_in_channels (int): Channels of concatenated depth+normal. Default: 4.
        norm_cfg (dict): Norm config for ConvModule. Default: None.
        act_cfg (dict): Act config for ConvModule. Default: None.
        fusion_type (str): 'gate' (default) uses spatial + channel gating, or
            'film' to apply FiLM-style affine modulation.
    """

    def __init__(
        self,
        in_channels,
        dn_in_channels: int = 4,
        norm_cfg=None,
        act_cfg=None,
        fusion_type: str = 'gate',
    ):
        super().__init__()
        assert isinstance(in_channels, (list, tuple)) and len(in_channels) == 4, \
            "in_channels must be a list/tuple of 4 integers"
        self.in_channels = list(in_channels)
        self.dn_in_channels = int(dn_in_channels)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        assert fusion_type in ('gate', 'film')
        self.fusion_type = fusion_type

        # Project DN to each level's channel dim after resizing to that level
        self.dn_proj = nn.ModuleList([
            ConvModule(
                in_channels=self.dn_in_channels,
                out_channels=ch,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ) for ch in self.in_channels
        ])

        # Spatial attention derived from DN guidance
        self.spa_att = nn.ModuleList([
            ConvModule(
                in_channels=ch,
                out_channels=1,
                kernel_size=3,
                padding=1,
                norm_cfg=None,
                act_cfg=None,
            ) for ch in self.in_channels
        ])

        # Channel attention (squeeze-excitation) from DN guidance
        self.cha_att = nn.ModuleList([
            _ChannelGate(channels=ch, reduction=4) for ch in self.in_channels
        ])

        # Optional FiLM parameters per level (gamma/beta from DN guidance)
        if self.fusion_type == 'film':
            self.film_gamma = nn.ModuleList([
                nn.Conv2d(ch, ch, kernel_size=1) for ch in self.in_channels
            ])
            self.film_beta = nn.ModuleList([
                nn.Conv2d(ch, ch, kernel_size=1) for ch in self.in_channels
            ])

        # Output smoothing per level
        self.smooth = nn.ModuleList([
            ConvModule(
                in_channels=ch,
                out_channels=ch,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ) for ch in self.in_channels
        ])

    def _parse_inputs(self, inputs):
        """Return (rgb_feats_list, dn) strictly from tensors.

        Accepts one of the following input formats:
          - dict with keys {'rgb_feats': [x1..x4], 'depth': Tensor(B,1,H,W), 'normal': Tensor(B,3,H,W)}
          - list/tuple of length 6: [x1, x2, x3, x4, depth, normal]
          - list/tuple of length 5: [x1, x2, x3, x4, dn] where dn is concatenated (B,4,H,W)
        """
        if isinstance(inputs, dict):
            rgb_feats = inputs.get('rgb_feats', inputs.get('rgb'))
            depth = inputs.get('depth', None)
            normal = inputs.get('normal', None)
            assert isinstance(rgb_feats, (list, tuple)) and len(rgb_feats) == 4, \
                "'rgb_feats' must be a list/tuple of 4 tensors"
            assert (depth is not None) and (normal is not None), \
                "dict input must provide both 'depth' and 'normal' tensors"
            dn = torch.cat([depth, normal], dim=1)
            return rgb_feats, dn
        else:
            assert isinstance(inputs, (list, tuple)), \
                "inputs must be list/tuple or dict"
            if len(inputs) == 6:
                rgb_feats = list(inputs[:4])
                depth, normal = inputs[4], inputs[5]
                dn = torch.cat([depth, normal], dim=1)
                return rgb_feats, dn
            elif len(inputs) == 5:
                rgb_feats = list(inputs[:4])
                dn = inputs[4]
                assert dn.dim() == 4 and dn.size(1) == self.dn_in_channels, \
                    f"Expected concatenated dn with {self.dn_in_channels} channels"
                return rgb_feats, dn
            else:
                raise AssertionError(
                    "DepthNormalFusionNeck expects 6 inputs (4 rgb + depth + normal) "
                    "or 5 inputs (4 rgb + concatenated dn), or a dict with keys.")

    def forward(self, inputs):
        rgb_feats, dn = self._parse_inputs(inputs)
        assert len(rgb_feats) == 4 and all(isinstance(x, torch.Tensor) for x in rgb_feats)

        outs = []
        B = rgb_feats[0].shape[0]
        for i in range(4):
            x = rgb_feats[i]
            # Prepare DN for this level
            if isinstance(dn, list):
                # dn is list of (1,4,h,w) possibly different sizes; resize each to x spatial size
                dn_resized_list = []
                for b in range(B):
                    dn_b = dn[b]
                    dn_b = F.interpolate(dn_b, size=x.shape[2:], mode='bilinear', align_corners=False)
                    dn_resized_list.append(dn_b)
                dn_resized = torch.cat(dn_resized_list, dim=0)  # (B,4,H,W)
            else:
                # batched tensor
                dn_resized = resize(dn, size=x.shape[2:], mode='bilinear', align_corners=False)
            dn_resized = dn_resized.to(x.device)

            # Project to match channels
            g = self.dn_proj[i](dn_resized)

            # Spatial attention from geometry
            spa = torch.sigmoid(self.spa_att[i](g))  # (B,1,H,W)
            # Channel attention from geometry
            cha = self.cha_att[i](g)                 # (B,C,1,1) in [0,1]

            if self.fusion_type == 'film':
                # FiLM affine conditioned on geometry
                gamma = self.film_gamma[i](F.adaptive_avg_pool2d(g, 1))
                beta = self.film_beta[i](F.adaptive_avg_pool2d(g, 1))
                y = gamma * x + beta
            else:
                y = x

            # Geometry-guided enhancement
            y = y + x * spa + x * cha + g
            y = self.smooth[i](y)
            outs.append(y)

        return tuple(outs)