python demo.py \
    ../../example/frame/frame.jpg \
    local_configs/segman/tiny/segman_t_cityscapes.py \
    checkpoint/segman_t_cityscapes.pth \
    --palette 'cityscapes' \
    --out-file show/segman_demo_t.png \
    --device 'cuda:0'