# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import torch
import cv2
import numpy as np
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from sklearn.decomposition import PCA
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def init_segmentor_efs(config, checkpoint=None, device='cuda:0', CLASSES=None,PALETTE=None ):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        try:
            model.CLASSES = checkpoint['meta']['CLASSES']
            model.PALETTE = checkpoint['meta']['PALETTE']
        except:
            model.CLASSES = CLASSES
            model.PALETTE = PALETTE
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def inference_segmentor_efs(model, img, depth=None, normal=None, mode="pca"):
    """Inference with multi-modal RGB-D-Normal input (7-channel).

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or ndarray): Image file or loaded image.
        depth (ndarray, optional): Depth map (HxW, float32).
        normal (ndarray, optional): Normal map (HxWx3, float32).
        mode (str): 融合方式，可选 ["rgbd", "rgbn", "fusion"]
            - "rgbd": 用 depth 替换 B 通道
            - "rgbn": 用 normal 替换 RGB
            - "fusion": 把 RGB+depth+normal 融合后做线性压缩到3通道
            - "svd": SVD 手写版，将 RGB+depth+normal 压缩到3通道
            - "pca": sklearn.PCA 版，将 RGB+depth+normal 压缩到3通道
            - "edge": 用 depth/normal 边缘增强 RGB
            - "hsv": 在 HSV 
            - "attn": attention 融合，根据 depth 边界决定几何信息权重

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build pipeline (only RGB part)
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # prepare RGB data
    data = dict(img=img)
    data = test_pipeline(data)  # after this, data['img'] is (C,H,W)

    # 转换成 numpy 以便和 depth/normal 融合
    rgb = data['img'][0].numpy().transpose(1, 2, 0)  # (H,W,3), float32
    H, W, _ = rgb.shape

    # resize depth/normal
    if depth is not None:
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)  # 归一化
        depth = depth[..., None]

    if normal is not None:
        if normal.shape[:2] != (H, W):
            normal = cv2.resize(normal, (W, H), interpolation=cv2.INTER_NEAREST)
        normal = (normal + 1.0) / 2.0          # 归一化到 [0,1]

    # ---- 融合策略 ----
    if mode == "rgbd" and depth is not None:
        # 用 depth 替换蓝色通道
        fused = rgb.copy()
        fused[..., 2] = depth[..., 0]

    elif mode == "rgbn" and normal is not None:
        # 直接用 normal 替代 RGB
        fused = normal

    elif mode == "fusion" and (depth is not None or normal is not None):
        # 融合 RGB+depth+normal，然后线性压缩到3通道
        feats = [rgb]
        if depth is not None:
            feats.append(np.repeat(depth, 3, axis=2))  # (H,W,3)
        if normal is not None:
            feats.append(normal)
        feats = np.concatenate(feats, axis=2)  # (H,W,C)

        # 简单线性压缩: (H,W,C) @ (C,3) -> (H,W,3)
        C = feats.shape[2]
        Wmat = np.random.randn(C, 3).astype(np.float32)
        fused = np.tensordot(feats, Wmat, axes=([2],[0]))
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    elif mode == "svd" and (depth is not None or normal is not None):
        feats = [rgb]
        if depth is not None:
            feats.append(np.repeat(depth, 3, axis=2))
        if normal is not None:
            feats.append(normal)
        X = np.concatenate(feats, axis=2)  # (H,W,C)
        H, W, C = X.shape
        X = X.reshape(-1, C)  # (H*W,C)

        # SVD 分解
        U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        X_svd = U[:, :3] @ np.diag(S[:3])
        fused = X_svd.reshape(H, W, 3)
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    elif mode == "pca" and (depth is not None or normal is not None):
        feats = [rgb]
        if depth is not None:
            feats.append(np.repeat(depth, 3, axis=2))
        if normal is not None:
            feats.append(normal)
        X = np.concatenate(feats, axis=2)  # (H,W,C)
        H, W, C = X.shape
        X = X.reshape(-1, C)  # (H*W,C)

        # sklearn PCA
        pca = PCA(n_components=3, whiten=False, random_state=42)
        X_pca = pca.fit_transform(X)  # (H*W,3)
        fused = X_pca.reshape(H, W, 3)
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    elif mode == "edge" and (depth is not None or normal is not None):
        # 用 depth/normal 边缘增强
        edge = np.zeros((H, W), np.float32)
        if depth is not None:
            d_edge = cv2.Canny((depth[..., 0] * 255).astype(np.uint8), 50, 150) / 255.0
            edge = np.maximum(edge, d_edge)
        if normal is not None:
            n_gray = cv2.cvtColor((normal * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            n_edge = cv2.Canny(n_gray, 50, 150) / 255.0
            edge = np.maximum(edge, n_edge)
        edge = cv2.GaussianBlur(edge, (5, 5), 1.0)[..., None]
        fused = np.clip(rgb + edge, 0, 1)

    elif mode == "hsv" and depth is not None:
        # 在 HSV 中用 depth 替代 V
        hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        hsv[..., 2] = depth[..., 0]
        fused = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    elif mode == "attn" and (depth is not None or normal is not None):
        # 简单 attention 融合
        feats = np.zeros_like(rgb)
        if depth is not None:
            feats += np.repeat(depth, 3, axis=2)
        if normal is not None:
            feats += normal
        # 权重 = depth 边缘强度
        if depth is not None:
            grad = cv2.Laplacian(depth[..., 0], cv2.CV_32F)
            alpha = 1 / (1 + np.abs(grad))
            alpha = np.clip(alpha, 0, 1)[..., None]
        else:
            alpha = 0.5
        fused = alpha * rgb + (1 - alpha) * feats
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    else:
        fused = rgb  # 默认只用RGB

    # 替换 data['img']，保持 (3,H,W)
    fused = fused.transpose(2, 0, 1).astype(np.float32)  # (3,H,W)
    fused = torch.from_numpy(fused)
    data['img'] = [fused]
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def show_result_pyplot_efs(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True,
                       out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    # plt.show(block=block)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img