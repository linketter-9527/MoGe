python moge/scripts/infer.py \
    -i example/side_frames \
    -o output/side_frames \
    --pretrained weights/moge-2-vitl-normal.pt \
    --device cuda:0 \
    --maps \
    --glb \
    --ply