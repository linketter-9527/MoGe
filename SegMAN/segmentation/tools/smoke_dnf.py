import argparse
import os
import sys

import torch
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.models import build_segmentor


def build_dummy_batch(H=256, W=512, B=2, device='cpu'):
    # RGB image
    img = torch.randn(B, 3, H, W, device=device)
    # Depth (1ch) and Normal (3ch)
    depth = torch.randn(B, 1, H, W, device=device)
    normal = torch.randn(B, 3, H, W, device=device)
    # Semantic GT (B,1,H,W), long dtype
    gt = torch.randint(low=0, high=19, size=(B, 1, H, W), dtype=torch.long, device=device)

    # Minimal img_metas per sample
    img_metas = []
    for _ in range(B):
        meta = dict(
            ori_shape=(H, W, 3),
            img_shape=(H, W, 3),
            pad_shape=(H, W, 3),
            scale_factor=1.0,
            flip=False,
            img_norm_cfg=dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        )
        img_metas.append(meta)
    return img, img_metas, gt, depth, normal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # Avoid depending on external pretrained weights for smoke test
    try:
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'backbone'):
            if isinstance(cfg.model.backbone, dict) and cfg.model.backbone.get('pretrained', None):
                cfg.model.backbone.pretrained = None
    except Exception:
        pass

    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    # SyncBN is not supported on CPU/DP in this standalone script; convert to BN
    model = revert_sync_batchnorm(model)

    model.to(args.device)
    model.train()  # forward_train path

    img, img_metas, gt, depth, normal = build_dummy_batch(args.height, args.width, B=2, device=args.device)

    print('Running forward_train with dummy batch ...')
    losses = model.forward_train(img, img_metas, gt, depth=depth, normal=normal)
    # Summarize loss scalars
    total = 0.0
    for k, v in losses.items():
        if torch.is_tensor(v):
            total = total + float(v.detach().mean().cpu())
        elif isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
            total = total + float(sum([_.detach().mean().cpu() for _ in v]))
    print('Loss keys:', list(losses.keys()))
    print('Total loss (approx):', total)

    print('SMOKE_OK')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('SMOKE_FAIL:', repr(e))
        sys.exit(1)