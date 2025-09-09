_base_ = './segman_t_bdd100k_dnf.py'

# override runner settings for smoke test
runner = dict(type='IterBasedRunner', max_iters=50)
checkpoint_config = dict(by_epoch=False, interval=10)
evaluation = dict(interval=10, metric='mIoU', save_best='mIoU')