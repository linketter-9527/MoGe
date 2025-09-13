python demo.py \
    ../../example/frame/bdd.jpg \
    local_configs/segman/tiny/segman_t_bdd100k.py \
    checkpoint/segman_t_bdd100k.pth \
    --palette 'bdd100k' \
    --out-file show/bdd100k/bdd_t.png \
    --device 'cuda:0'