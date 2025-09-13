python demo.py \
    ../../example/frame/bdd.jpg \
    local_configs/segman/tiny/segman_t_cityscapes.py \
    checkpoint/segman_t_cityscapes.pth \
    --palette 'cityscapes' \
    --out-file show/cityscapes/city_t.png \
    --device 'cuda:0'