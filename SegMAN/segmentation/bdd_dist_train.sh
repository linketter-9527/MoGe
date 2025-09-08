bash tools/dist_train.sh \
    local_configs/segman/tiny/segman_t_bdd100k.py \
    2 \
    --work-dir outputs/segman_t_bdd100k \
    --drop-path 0.0  # large = 0.3, base = 0.25, small = 0.2, tiny = 0.0