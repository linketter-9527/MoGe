_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/datasets/bdd100k_dnf_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderDNF',
    backbone=dict(
        type='SegMANEncoder_t',
        pretrained='pretrained/SegMAN_Encoder_t.pth.tar',
        style='pytorch'
    ),
    neck=dict(
        type='DepthNormalFusionNeck',
        in_channels=[32, 64, 144, 192],
        dn_in_channels=4,
        fusion_type='gate',
    ),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# data
data = dict(samples_per_gpu=1)  # total batch size depends on GPUs

# evaluation
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')