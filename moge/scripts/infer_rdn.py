import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import itertools
import json
import warnings

import time

import click

# 添加SegMAN相关的导入
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

@click.command(help='Inference script')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), help='Input image or folder path. "jpg" and "png" are supported.')
@click.option('--fov_x', 'fov_x_', type=float, default=None, help='If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.')
@click.option('--output', '-o', 'output_path', default='./output', type=click.Path(), help='Output folder path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default=None, help='Pretrained model name or path. If not provided, the corresponding default model will be chosen.')
@click.option('--version', 'model_version', type=click.Choice(['v1', 'v2']), default='v2', help='Model version. Defaults to "v2"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Use fp16 precision for much faster inference.')
@click.option('--resize', 'resize_to', type=int, default=None, help='Resize the image(s) & output maps to a specific size. Defaults to None (no resizing).')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
Higher value means more tokens and the finer details will be captured, but inference can be slower. \
Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size. \
`resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.')
@click.option('--num_tokens', type=int, default=None, help='number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`. \
`resolution_level` will be ignored if `num_tokens` is provided. Default: None')
@click.option('--threshold', type=float, default=0.04, help='Threshold for removing edges. Defaults to 0.01. Smaller value removes more edges. "inf" means no thresholding.')

# 添加SegMAN相关的命令行参数
@click.option('--seg-config', type=str, default=None, help='SegMAN config file path for semantic segmentation')
@click.option('--seg-checkpoint', type=str, default=None, help='SegMAN checkpoint file path for semantic segmentation')
@click.option('--seg-palette', type=str, default='cityscapes', help='Color palette used for segmentation map')
@click.option('--extract-target', type=int, default=0, help='extract target class number. Defaults to 0.')

@click.option('--maps', 'save_maps_', is_flag=True, help='Whether to save the output maps (image, point map, depth map, normal map, mask) and fov.')
@click.option('--glb', 'save_glb_', is_flag=True, help='Whether to save the output as a.glb file. The color will be saved as a texture.')
@click.option('--ply', 'save_ply_', is_flag=True, help='Whether to save the output as a.ply file. The color will be saved as vertex colors.')
@click.option('--seg', 'save_seg_', is_flag=True, help='Whether to save the output as semantic segmentation.')
@click.option('--edge', 'save_edge_', is_flag=True, help='Whether to save the output as extract edge.')
@click.option('--rdn', 'save_rdn_', is_flag=True, help='Whether to save the output as a.npy file. The rgb、depth、normal will be saved as npy.')
@click.option('--show', 'show', is_flag=True, help='Whether show the output in a window. Note that this requires pyglet<2 installed as required by trimesh.')

def main(
    input_path: str,
    fov_x_: float,
    output_path: str,
    pretrained_model_name_or_path: str,
    model_version: str,
    device_name: str,
    use_fp16: bool,
    resize_to: int,
    resolution_level: int,
    num_tokens: int,
    threshold: float,
    # 添加SegMAN参数
    seg_config: str,
    seg_checkpoint: str,
    seg_palette: str,
    extract_target: int,
    save_maps_: bool,
    save_glb_: bool,
    save_ply_: bool,
    save_seg_: bool,
    save_edge_: bool,
    save_rdn_: bool,
    show: bool,
):  
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    import trimesh
    import trimesh.visual
    import click

    from moge.model import import_model_class_by_version
    from moge.utils.io import save_glb, save_ply
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
    from moge.utils.extract_edge import extract_edge, visualize_edges
    from moge.utils.extract_target_edge import extract_target_edge, visualize_target_edges
    from moge.utils.extract_edge_from_depth_normal import extract_edge_from_depth_normal
    import utils3d

    import matplotlib.pyplot as plt

    device = torch.device(device_name)

    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [Path(input_path)]
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f'No image files found in {input_path}')

    if pretrained_model_name_or_path is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_version]
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    if use_fp16:
        model.half()
    
    if not any([save_maps_, save_glb_, save_ply_, save_seg_, save_edge_, save_rdn_]):
        warnings.warn('No output format specified. Defaults to saving all. Please use "--maps", "--glb", "--ply", "--seg", "--edge" or "--rdn" to specify the output.')
        save_maps_ = save_glb_ = save_ply_ = save_seg_ = save_edge_ = save_rdn_ = True

    # 初始化SegMAN语义分割模型（如果提供了配置和检查点）
    seg_model = None
    if seg_config and seg_checkpoint:
        try:
            seg_model = init_segmentor(seg_config, seg_checkpoint, device=device_name)
            # print(f"SegMAN semantic segmentation model loaded from {seg_checkpoint}")
        except Exception as e:
            print(f"Failed to load SegMAN model: {e}")
            seg_model = None
    
    # total_seg_time = 0.0
    # total_moge_time = 0.0
    # total_edge_time = 0.0

    save_path = Path(output_path)
    save_path.mkdir(exist_ok=True, parents=True)
    # depth_dir = save_path / "depth"
    # depth_dir.mkdir(exist_ok=True, parents=True)
    # normal_dir = save_path / "normal"
    # normal_dir.mkdir(exist_ok=True, parents=True)
    # mask_dir = save_path / "mask"
    # mask_dir.mkdir(exist_ok=True, parents=True)
    rdn_dir = save_path / "rdn"
    rdn_dir.mkdir(exist_ok=True, parents=True)
    meta_dir = save_path / "meta"
    meta_dir.mkdir(exist_ok=True, parents=True)


    for image_path in (pbar := tqdm(image_paths, desc='Inference', disable=len(image_paths) <= 1)):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        # save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        # save_path.mkdir(exist_ok=True, parents=True)

        file_prefix = f"{image_path.stem}_"
        file_noprefix = f"{image_path.stem}"
        # save_path = Path(output_path)
        # save_path.mkdir(exist_ok=True, parents=True)

        """
        seg_start_time = time.time()
        # 执行语义分割（如果SegMAN模型已加载）
        seg_result = None
        if seg_model is not None:
            try:
                seg_result = inference_segmentor(seg_model, str(image_path))
                # 显示或保存分割结果
                if save_seg_:
                    show_result_pyplot(seg_model, str(image_path), seg_result, get_palette(seg_palette), 
                                        out_file=str(save_path / f'{file_prefix}seg.png'), opacity=0.9, block=False)
                # print(f"Semantic segmentation completed for {image_path}")
            except Exception as e:
                print(f"Semantic segmentation failed for {image_path}: {e}")
        seg_elapsed = time.time() - seg_start_time
        print(f"[SegMAN] Semantic segmentation took {seg_elapsed:.3f} seconds.")
        total_seg_time += seg_elapsed
        """

        # moge_start_time = time.time()
        # MoGe2推理
        output = model.infer(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=use_fp16)
        points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        normal = output['normal'].cpu().numpy() if 'normal' in output else None
        # moge_elapsed = time.time() - moge_start_time
        # print(f"[MoGe] Inference took {moge_elapsed:.3f} seconds.")
        # total_moge_time += moge_elapsed

        # print(f"[Debug] depth:  shape: {depth.shape}, dtype: {depth.dtype}")
        # print(f"[Debug] normal:  shape: {normal.shape}, dtype: {normal.dtype}")

        """
        edge_start_time = time.time()
        geo_edge = extract_edge_from_depth_normal(depth, normal, mode="canny")

        # 保存单张边缘图，转换为8位灰度图 (0-255)
        # edge_8bit = (geo_edge * 255).astype(np.uint8)
        # cv2.imwrite(str(save_path / f'{file_prefix}edge.png'), edge_8bit)

        # 将分割结果转换为类别掩码（假设类别为0）
        # seg_mask = seg_result[extract_target].astype(np.uint8)
        # target_edge = extract_target_edge(seg_mask, geo_edge, class_id=extract_target, dilation_size=5)

        seg_mask = (seg_result[0] == extract_target).astype(np.uint8)  # 直接生成目标类别掩码
        """

        """
        plt.figure(figsize=(8, 6))
        plt.title("seg_mask")
        plt.imshow(seg_mask, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=True)
        """

        """
        target_edge = extract_target_edge(seg_mask, geo_edge, dilation_size=5)

        # cv2.imwrite(str(save_path / f'{file_prefix}target_edge.png'), target_edge)
        """

        """
        # 提取目标边沿
        # 将分割结果转换为类别掩码（假设类别为0）
        seg_mask = seg_result[extract_target].astype(np.uint8)
        # 提取目标边沿
        edges = extract_edge(
            normal_map=normal,  # 取第一个batch
            depth_map=depth,    # 取第一个batch
            segmentation_mask=seg_mask,
            class_id=extract_target,       # 根据实际分割类别调整
            theta_threshold=20.0,
            height_threshold=0.05
        )        
        """
        
        """
        edge_elapsed = time.time() - edge_start_time
        print(f"[edge] Extract edge took {edge_elapsed:.3f} seconds.")
        total_edge_time += edge_elapsed

        # 可视化目标边沿
        target_edge_visualization = visualize_target_edges(
            image=image,
            target_edge_mask=target_edge,
            color=(255, 0, 0),  # 红色边沿
            thickness=2
        )
        """

        if save_edge_:
            # 保存边沿可视化结果
            cv2.imwrite(str(save_path / f'{file_prefix}edges.png'), cv2.cvtColor(target_edge_visualization, cv2.COLOR_RGB2BGR))
                
            # 保存原始边沿掩码
            # cv2.imwrite(str(save_path / f'{file_prefix}edges_mask.png'), (edges * 255).astype(np.uint8))       

        # 保存 BGR + Depth + Normal 为 7 通道 .npy
        if save_rdn_:
            # RGB (H,W,3) -> BGR (H,W,3) uint8
            bgr_ch = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # depth (H,W,1) float16
            depth_ch = depth[..., None].astype(np.float16)
            # normal (H,W,3) float16
            normal_ch = normal.astype(np.float16)
            # 拼接成 (H, W, 7)
            rdn = np.concatenate([bgr_ch, depth_ch, normal_ch], axis=-1)
            
            # 保存.npy文件（禁止pickle）
            np.save(str(rdn_dir / f'{file_noprefix}.npy'), rdn, allow_pickle=False)
            
            # 保存元数据
            metadata = {
                'depth_dtype': 'float16',
                'normal_dtype': 'float16',
                'image_dtype': 'float16',
                'height': height,
                'width': width
            }
            with open(str(meta_dir / f'{file_prefix}metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            # time.sleep(0.2)

        # Save images / maps
        if save_maps_:
            # cv2.imwrite(str(save_path / f'{file_prefix}image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(str(save_path / f'{file_prefix}depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(depth_dir / f'{file_noprefix}.exr'), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(mask_dir / f'{file_noprefix}.png'), (mask * 255).astype(np.uint8))
            # cv2.imwrite(str(save_path / f'{file_prefix}points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            if normal is not None:
                # cv2.imwrite(str(save_path / f'{file_prefix}normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(normal_dir / f'{file_noprefix}.exr'), normal, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            """
            fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
            with open(save_path / f'{file_prefix}fov.json', 'w') as f:
                json.dump({
                    'fov_x': round(float(np.rad2deg(fov_x)), 2),
                    'fov_y': round(float(np.rad2deg(fov_y)), 2),
                }, f)
            """

        # Export mesh网格 & visulization
        if save_glb_ or save_ply_ or show:
            mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=threshold)
            if normal is None:
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=mask_cleaned,
                    tri=True
                )
                vertex_normals = None
            else:
                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    normal,
                    mask=mask_cleaned,
                    tri=True
                )
            # 导出时遵循OpenGL坐标约定When exporting the model, follow the OpenGL coordinate conventions:
            # - world coordinate system: x right, y up, z backward.
            # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
            vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
            if normal is not None:
                vertex_normals = vertex_normals * [1, -1, -1]

        if save_glb_:
            save_glb(save_path / f'{file_prefix}mesh.glb', vertices, faces, vertex_uvs, image, vertex_normals)

        if save_ply_:
            save_ply(save_path / f'{file_prefix}pointcloud.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

        if show:
            trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                vertex_normals=vertex_normals,
                faces=faces, 
                process=False
            ).show()  

    # print(f"Total SegMAN time: {total_seg_time:.3f} seconds.")
    # print(f"Total MoGe time: {total_moge_time:.3f} seconds.")
    # print(f"Total edge time: {total_edge_time:.3f} seconds.")

if __name__ == '__main__':
    main()
