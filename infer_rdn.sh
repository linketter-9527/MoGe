python moge/scripts/infer_rdn.py \
    -i example/train \
    -o output/train \
    --pretrained weights/moge-2-vitl-normal.pt \
    --device cuda:0 \
    --seg-config SegMAN/segmentation/local_configs/segman/large/segman_l_cityscapes.py \
    --seg-checkpoint SegMAN/segmentation/checkpoint/segman_l_cityscapes.pth \
    --seg-palette 'cityscapes' \
    --extract-target 0 \
    --maps
    # --maps \
    # --glb \
    # --ply \
    # --seg \
    # --edge \
    # --rdn