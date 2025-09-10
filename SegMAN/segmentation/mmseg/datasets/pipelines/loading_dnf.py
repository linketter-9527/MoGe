# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFileDNF(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotationsDNF(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class AddDepthNormalPathsDNF(object):
    """Construct depth/normal file paths from image filename.

    This transform assumes depth and normal files share the same relative
    filename as the RGB image but may have different suffixes and reside in
    different root directories.

    Args:
        depth_root (str, optional): Root directory for depth files.
        normal_root (str, optional): Root directory for normal files.
        depth_suffix (str): Suffix/extension for depth files. Default: '.exr'.
        normal_suffix (str): Suffix/extension for normal files. Default: '.exr'.
    """

    def __init__(self,
                 depth_root=None,
                 normal_root=None,
                 depth_suffix='.exr',
                 normal_suffix='.exr'):
        self.depth_root = depth_root
        self.normal_root = normal_root
        self.depth_suffix = depth_suffix
        self.normal_suffix = normal_suffix

    def __call__(self, results):
        img_rel = results['img_info']['filename']
        base, _ = osp.splitext(img_rel)
        if self.depth_root is not None:
            results['depth_path'] = osp.join(self.depth_root, base + self.depth_suffix)
        if self.normal_root is not None:
            results['normal_path'] = osp.join(self.normal_root, base + self.normal_suffix)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(depth_root={self.depth_root}, "
                f"normal_root={self.normal_root}, depth_suffix='{self.depth_suffix}', "
                f"normal_suffix='{self.normal_suffix}')")


@PIPELINES.register_module()
class LoadNpyDepthNormalFromFileDNF(object):
    """Load a 7-channel npy (BGR float16 + depth float16 + normal float16) and split into img/depth/normal.

    Required keys: img_prefix (optional), img_info['filename']. 
    Added keys: 'img' (H,W,3 uint8), 'depth' (H,W float32), 'normal' (H,W,3 float32),
                and common meta fields similar to LoadImageFromFileDNF.
    """

    def __init__(self, file_client_args=dict(backend='disk')):
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        # Use numpy to load the packed 7-channel array
        arr = np.load(filename)
        assert arr.ndim == 3 and arr.shape[2] == 7, \
            f"Expect HxWx7 npy, got shape {arr.shape} at {filename}"
        # BGR channels are float16; convert to uint8 expected by Normalize(mean/std)
        img = arr[:, :, :3].astype(np.uint8)  # Direct conversion from float16 to uint8
        # Depth channel keep raw values (may contain +inf)
        depth = arr[:, :, 3].astype(np.float32)
        depth = np.where(np.isfinite(depth), depth, 1024.0)
        # Normal channels float16 -> float32; keep raw values (no re-normalization here)
        normal = arr[:, :, 4:7].astype(np.float32)
        # Keep original normal values (no re-normalization); invalid regions will be masked later in the neck
        # norm = np.linalg.norm(normal, axis=2, keepdims=True)
        # norm = np.maximum(norm, 1e-6)
        # normal = normal / norm

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Common meta
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
            to_rgb=False)
        # Geometry payloads
        results['depth'] = depth
        results['normal'] = normal
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(file_client_args={self.file_client_args})"


@PIPELINES.register_module()
class ResizeGeometryToMatch(object):
    """Resize depth and normal to match current image size (results['img']).

    Should be placed right after any spatial resize/op applied to 'img'.
    - depth: bilinear interpolation
    - normal: bilinear per-channel (no re-normalization)
    """

    def __init__(self, align_corners=False):
        self.align_corners = align_corners

    def __call__(self, results):
        h, w = results['img'].shape[:2]
        if 'depth' in results:
            d = results['depth']
            if d.shape[0] != h or d.shape[1] != w:
                # cv2 resize expects (W,H)
                results['depth'] = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        if 'normal' in results:
            n = results['normal']
            if n.shape[0] != h or n.shape[1] != w:
                n_resized = np.zeros((h, w, 3), dtype=np.float32)
                for c in range(3):
                    n_resized[:, :, c] = cv2.resize(n[:, :, c], (w, h), interpolation=cv2.INTER_LINEAR)
                # no re-normalization to preserve original normals' magnitude/direction as provided
                # norm = np.linalg.norm(n_resized, axis=2, keepdims=True)
                # norm = np.maximum(norm, 1e-6)
                # n_resized = n_resized / norm
                results['normal'] = n_resized

        # update meta shapes to image if needed
        results['img_shape'] = results['img'].shape
        results['pad_shape'] = results['img'].shape
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(align_corners={self.align_corners})"


@PIPELINES.register_module()
class PadGeometryToMatch(object):
    """Pad depth/normal to the same spatial size as results['pad_shape'] after Pad.

    This should be placed right after the Pad transform on 'img'.
    """

    def __init__(self, pad_val_depth=0.0, pad_val_normal=0.0):
        self.pad_val_depth = float(pad_val_depth)
        self.pad_val_normal = float(pad_val_normal)

    def __call__(self, results):
        target_h, target_w = results['pad_shape'][:2]
        h, w = results['img'].shape[:2]
        assert target_h >= h and target_w >= w
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return results
        # pad depth
        if 'depth' in results:
            d = results['depth']
            dh, dw = d.shape[:2]
            top = 0
            bottom = target_h - dh
            left = 0
            right = target_w - dw
            if bottom > 0 or right > 0:
                results['depth'] = cv2.copyMakeBorder(
                    d, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                    value=self.pad_val_depth)
        # pad normal (3 channels)
        if 'normal' in results:
            n = results['normal']
            nh, nw = n.shape[:2]
            top = 0
            bottom = target_h - nh
            left = 0
            right = target_w - nw
            if bottom > 0 or right > 0:
                results['normal'] = cv2.copyMakeBorder(
                    n, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                    value=self.pad_val_normal)

        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(pad_val_depth={self.pad_val_depth}, "
                f"pad_val_normal={self.pad_val_normal})")
