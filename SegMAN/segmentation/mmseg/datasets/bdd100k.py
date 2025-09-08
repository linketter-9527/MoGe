# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class BDD100KDataset(CustomDataset):
    CLASSES = (  # from bdd100k 19 classes
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
        'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 depth_dir=None,
                 normal_dir=None,
                 data_root=None,
                 **kwargs):
        # Note: data_root is also accepted by CustomDataset. We pass it down
        super(BDD100KDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            data_root=data_root,
            **kwargs)
        # Additional dirs for depth/normal
        self.depth_dir = depth_dir
        self.normal_dir = normal_dir
        # join with data_root if provided and not absolute
        if self.data_root is not None:
            if self.depth_dir is not None and not osp.isabs(self.depth_dir):
                self.depth_dir = osp.join(self.data_root, self.depth_dir)
            if self.normal_dir is not None and not osp.isabs(self.normal_dir):
                self.normal_dir = osp.join(self.data_root, self.normal_dir)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline and add depth/normal prefixes."""
        super().pre_pipeline(results)
        if getattr(self, 'depth_dir', None) is not None:
            results['depth_prefix'] = self.depth_dir
        if getattr(self, 'normal_dir', None) is not None:
            results['normal_prefix'] = self.normal_dir