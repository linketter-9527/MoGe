import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
from skimage.morphology import skeletonize

def extract_target_edge(
    target_mask: np.ndarray,
    geo_edge: np.ndarray,
    # class_id: int = 0,
    dilation_size: int = 5,
    visualize: bool = False,   # 可选可视化开关
) -> np.ndarray:
    """
    参数:
        seg_mask: 语义分割目标掩码
        class_id: 目标在分割掩码中的类别ID
        dilation_size: 形态学膨胀核大小
        geo_edge: 来自 MoGe 模型生成的边缘图
    """
    # 1. 从语义分割中提取目标掩码
    # target_mask = (seg_mask == class_id).astype(np.uint8)

    # 2. 使用形态学操作找到目标边界候选区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    dilated_target = cv2.dilate(target_mask, kernel, iterations=1)
    eroded_target = cv2.erode(target_mask, kernel, iterations=1)
    
    # 边界候选区域：膨胀后的目标与腐蚀后的目标的差异区域
    boundary_candidate = dilated_target - eroded_target
    # dilated_target - eroded_target

    # 额外膨胀一次，让边界更粗
    kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    boundary_candidate = cv2.dilate(boundary_candidate, kernel_boundary, iterations=1)

    # 确保boundary_candidate为二值图
    boundary_candidate = boundary_candidate.astype(np.uint8)

    """
    # ====== 可视化部分 ======
    if visualize:
        plt.figure(figsize=(8, 6))
        plt.title("Boundary Candidate")
        plt.imshow(boundary_candidate, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=True)   # 阻塞显示，手动关闭窗口后才继续执行
    """

    # 使用边界候选区域作为掩码，只保留geo_edge中在候选区域内的边缘
    # target_edge = cv2.bitwise_and(geo_edge, geo_edge, mask=boundary_candidate)
    target_edge = geo_edge * (boundary_candidate > 0).astype(geo_edge.dtype)

    # === 1. 细化 skeletonize ===
    skeleton = skeletonize(target_edge > 0)
    target_edge = (skeleton.astype(np.uint8)) * 255

    # === 2. 连通（闭运算连接断裂）===
    kernel_conn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (125, 125))
    target_edge = cv2.morphologyEx(target_edge, cv2.MORPH_CLOSE, kernel_conn)

    """
    # === 3. 去噪（连通域分析，去除小块）===
    target_edge = (target_edge > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(target_edge, connectivity=8)
    min_area = 50  # 面积阈值，可调
    filtered = np.zeros_like(target_edge)
    for i in range(1, num_labels):  # 跳过背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    target_edge = filtered
    """

    skeleton = skeletonize(target_edge > 0)
    target_edge = (skeleton.astype(np.uint8)) * 255

    if visualize:
        plt.figure(figsize=(24, 6))
        plt.subplot(1, 3, 1)
        plt.title("Boundary Candidate")
        plt.imshow(boundary_candidate, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Raw Target Edge")
        plt.imshow(geo_edge, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Final Target Edge")
        plt.imshow(target_edge, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show(block=True)

    return target_edge

def visualize_target_edges(
    image: np.ndarray,
    target_edge_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上可视化道路边沿
    
    参数:
        image: 原始图像 (H, W, 3)
        target_edge_mask: 道路边沿掩码 (H, W)
        color: 边沿颜色 (B, G, R)
        thickness: 边沿线粗细
        
    返回:
        可视化结果图像
    """
    result = image.copy()
    
    # 找到边沿点的坐标
    y_coords, x_coords = np.where(target_edge_mask > 0)
    
    # 在图像上绘制边沿点
    for y, x in zip(y_coords, x_coords):
        cv2.circle(result, (x, y), thickness, color, -1)
    
    return result
