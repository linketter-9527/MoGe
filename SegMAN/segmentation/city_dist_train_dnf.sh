bash tools/dist_train_dnf.sh \
    local_configs/segman/tiny/segman_t_cityscapes_dnf.py \
    2 \
    --work-dir outputs/segman_2bs_dp0.0_t_city_dnf_c1500 \
    --drop-path 0.0  # large = 0.3, base = 0.25, small = 0.2, tiny = 0.0