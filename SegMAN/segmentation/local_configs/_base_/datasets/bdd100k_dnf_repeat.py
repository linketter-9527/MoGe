# dataset settings
dataset_type = 'BDD100KDataset'
data_root = 'data/bdd100k/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadNpyDepthNormalFromFileDNF'),
    dict(type='LoadAnnotationsDNF'),
    # 统一到固定尺寸，避免随机裁剪/翻转带来的几何对齐问题
    dict(type='Resize', img_scale=(2048, 1024), keep_ratio=False),
    dict(type='ResizeGeometryToMatch'),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  # removed
    # dict(type='RandomFlip', prob=0.5),  # removed
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # 使用 size_divisor 对齐到 32 的倍数，避免固定裁剪尺寸带来的 shape 约束
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
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeGeometryToMatch'),
            # dict(type='RandomFlip'),  # removed for geometry alignment
            dict(type='Normalize', **img_norm_cfg),
            # 推理同样对齐到32的倍数，保持与训练一致，并同步填充几何数据
            dict(type='Pad', size_divisor=32),
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
            img_dir='images/10k/train',
            img_suffix='.npy',
            ann_dir='labels/sem_seg/masks/train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/10k/val',
        img_suffix='.npy',
        ann_dir='labels/sem_seg/masks/val',
            pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/10k/val',
        img_suffix='.npy',
        ann_dir='labels/sem_seg/masks/val',
            pipeline=test_pipeline))