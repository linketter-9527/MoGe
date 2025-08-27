python moge/scripts/infer_ref.py \
    -i example/frame \
    -o output/frames \
    --pretrained weights/moge-2-vitl-normal.pt \
    --device cuda:0 \
    --seg-config SegMAN/segmentation/local_configs/segman/large/segman_l_cityscapes_ref.py \
    --seg-checkpoint SegMAN/segmentation/checkpoint/segman_l_cityscapes.pth \
    --seg-palette 'cityscapes' \
    --seg
    # --extract-target 0 \
    # --edge \
    # --maps \
    # --glb \
    # --ply \
    # --seg 