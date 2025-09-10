# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter, DataContainer as DC
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.datasets.pipelines.formatting_dnf import to_tensorDNF
from mmseg.models import build_segmentor


def init_segmentor_dnf(config, checkpoint=None, device='cuda:0', CLASSES=None,PALETTE=None ):
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
    """A simple pipeline to load image (expects ndarray)."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the ndarray image.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        img = results['img']
        if not isinstance(img, np.ndarray):
            raise TypeError(f'LoadImage 期望 numpy.ndarray，但得到 {type(img)}')
        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor_dnf(model, img, depth=None, normal=None):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        depth (np.ndarray or torch.Tensor, optional): Single-image depth map.
            Accepts shape (H,W), (1,H,W) or (H,W,1). Will be converted to (1,1,H,W).
        normal (np.ndarray or torch.Tensor, optional): Single-image normal map.
            Accepts shape (H,W,3), (3,H,W) or (1,3,H,W). Will be converted to (1,3,H,W).

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    pipeline_rest = cfg.data.test.pipeline[1:]
    # 当提供了内存中的几何张量时，移除依赖文件名的 AddDepthNormalPathsDNF 和 LoadNpyDepthNormalFromFileDNF
    if depth is not None and normal is not None:
        def _is_dn_related(t):
            return (isinstance(t, dict) and 
                   (t.get('type') == 'AddDepthNormalPathsDNF' or 
                    t.get('type') == 'LoadNpyDepthNormalFromFileDNF'))
        pipeline_rest = [t for t in pipeline_rest if not _is_dn_related(t)]
    test_pipeline = [LoadImage()] + pipeline_rest
    test_pipeline = Compose(test_pipeline)
    # prepare data (single image)
    img_data = test_pipeline({'img': img})
    
    # 当提供了内存中的几何张量时，手动添加并格式化depth和normal到结果中
    if depth is not None and normal is not None:
        # 处理depth数据：统一转换为(C, H, W)格式的DataContainer
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()
        
        # 统一depth格式为(H, W, C)
        if depth.ndim == 2:  # (H, W) -> (H, W, 1)
            depth = depth[:, :, None]
        elif depth.ndim == 3 and depth.shape[0] == 1:  # (1, H, W) -> (H, W, 1)
            depth = depth.transpose(1, 2, 0)
        
        # 转换为(C, H, W)并包装为DataContainer
        depth = np.ascontiguousarray(depth.transpose(2, 0, 1)).astype(np.float32)
        img_data['depth'] = DC(to_tensorDNF(depth), stack=True)
        
        # 处理normal数据：统一转换为(C, H, W)格式的DataContainer
        if torch.is_tensor(normal):
            normal = normal.cpu().numpy()
        
        # 统一normal格式为(3, H, W)
        if normal.ndim == 3 and normal.shape[2] == 3:  # (H, W, 3) -> (3, H, W)
            normal = normal.transpose(2, 0, 1)
        elif normal.ndim == 4 and normal.shape[0] == 1:  # (1, 3, H, W) -> (3, H, W)
            normal = normal[0]
        
        # 包装为DataContainer
        normal = np.ascontiguousarray(normal).astype(np.float32)
        img_data['normal'] = DC(to_tensorDNF(normal), stack=True)
    
    data = collate([img_data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU，获取设备索引（整数）而不是torch.device对象
        device_index = device.index if device.index is not None else 0
        scattered_data = scatter(data, [device_index])
        if isinstance(scattered_data, list):
            data = scattered_data[0]
        else:
            data = scattered_data
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def show_result_pyplot_dnf(model,
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