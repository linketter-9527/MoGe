python moge/scripts/infer_dnf.py \
    -i example/frame \
    -o output/dnf/frame \
    --pretrained weights/moge-2-vitl-normal.pt \
    --device cuda:0 \
    --seg-config SegMAN/segmentation/local_configs/segman/tiny/segman_t_bdd100k_dnf.py \
    --seg-checkpoint SegMAN/segmentation/checkpoint/segman_2bs_dp0.0_t_bdd100k_dnf.pth \
    --seg-palette 'bdd100k' \
    --extract-target 0 \
    --seg
    # --edge \
    # --maps \
    # --glb \
    # --ply \
    # --seg 