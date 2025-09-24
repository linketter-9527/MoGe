bash tools/dist_train.sh \
    local_configs/segman/tiny/segman_t_cityscapes.py \
    2 \
    --work-dir outputs/segman_4bs_dp0.0_t_cityscapes \
    --drop-path 0.0  # large = 0.3, base = 0.25, small = 0.2, tiny = 0.0