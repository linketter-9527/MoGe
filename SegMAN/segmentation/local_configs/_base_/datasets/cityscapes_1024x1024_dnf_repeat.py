# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes_dnf/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadNpyDepthNormalFromFileDNF'),
    dict(type='LoadAnnotationsDNF'),
    dict(type='Resize', img_scale=(2048, 1024), keep_ratio=False),
    dict(type='ResizeGeometryToMatch'),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PadGeometryToMatch'),
    dict(type='DefaultFormatBundleDNF'),
    dict(type='CollectDNF', keys=['img', 'gt_semantic_seg', 'depth', 'normal']),
]
test_pipeline = [
    dict(type='LoadNpyDepthNormalFromFileDNF'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeGeometryToMatch'),
            # dict(type='RandomFlip'),  # removed for geometry alignment
            dict(type='Normalize', **img_norm_cfg),
            dict(type='PadGeometryToMatch'),
            dict(type='DefaultFormatBundleDNF'),
            dict(type='CollectDNF', keys=['img', 'depth', 'normal']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='leftImg8bit/train',
            img_suffix='.npy',
            ann_dir='gtFine/train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        img_suffix='.npy',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        img_suffix='.npy',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
