import numpy as np
import cv2
from typing import Tuple, Optional

def extract_edge(
    normal_map: np.ndarray,
    depth_map: np.ndarray,
    segmentation_mask: np.ndarray,
    class_id: int = 0,
    theta_threshold: float = 20.0,  # 法向夹角阈值（度）
    height_threshold: float = 0.05,  # 高度差阈值（米）
    dilation_size: int = 3,
    min_edge_length: int = 10
) -> np.ndarray:
    """
    结合法向量和语义分割信息检测道路边沿
    
    参数:
        normal_map: 法向量图 (H, W, 3)
        depth_map: 深度图 (H, W)
        segmentation_mask: 语义分割掩码 (H, W)，目标类别为class_id
        class_id: 目标在分割掩码中的类别ID
        theta_threshold: 法向夹角阈值（度）
        height_threshold: 高度差阈值（米）
        dilation_size: 形态学膨胀核大小
        min_edge_length: 最小边沿长度
        
    返回:
        边沿二值掩码 (H, W)
    """
    # 1. 从语义分割中提取目标掩码
    target_mask = (segmentation_mask == class_id).astype(np.uint8)
    
    # 2. 使用形态学操作找到目标边界候选区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    dilated_target = cv2.dilate(target_mask, kernel, iterations=1)
    eroded_target = cv2.erode(target_mask, kernel, iterations=1)
    
    # 边界候选区域：膨胀后的目标与腐蚀后的目标的差异区域
    boundary_candidate = dilated_target - eroded_target
    
    # 3. 在边界候选区域内计算法向差异和高度梯度
    edge_mask = np.zeros_like(target_mask, dtype=np.uint8)
    
    # 只处理边界候选区域内的像素
    y_coords, x_coords = np.where(boundary_candidate > 0)
    
    for y, x in zip(y_coords, x_coords):
        if x == 0 or y == 0 or x == normal_map.shape[1]-1 or y == normal_map.shape[0]-1:
            continue
            
        # 计算法向差异
        current_normal = normal_map[y, x]
        
        # 检查4邻域的法向差异
        neighbors = [
            (y-1, x), (y+1, x), (y, x-1), (y, x+1),
            (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)
        ]
        
        max_angle_diff = 0.0
        for ny, nx in neighbors:
            if 0 <= ny < normal_map.shape[0] and 0 <= nx < normal_map.shape[1]:
                neighbor_normal = normal_map[ny, nx]
                # 计算法向夹角（度）
                dot_product = np.dot(current_normal, neighbor_normal)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle_diff = np.degrees(np.arccos(dot_product))
                max_angle_diff = max(max_angle_diff, angle_diff)
        
        # 计算高度梯度
        current_depth = depth_map[y, x]
        depth_diffs = []
        for ny, nx in neighbors:
            if 0 <= ny < depth_map.shape[0] and 0 <= nx < depth_map.shape[1]:
                depth_diffs.append(abs(current_depth - depth_map[ny, nx]))
        
        max_height_diff = max(depth_diffs) if depth_diffs else 0.0
        
        # 判断是否为边沿点
        if (max_angle_diff > theta_threshold and 
            max_height_diff > height_threshold):
            edge_mask[y, x] = 1
    
    # 4. 后处理：去除孤立点，保持连续曲线
    # 先进行形态学闭操作连接断点
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    # 去除小连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_edge_length:
            edge_mask[labels == i] = 0
    
    return edge_mask

def visualize_edges(
    image: np.ndarray,
    edge_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上可视化道路边沿
    
    参数:
        image: 原始图像 (H, W, 3)
        edge_mask: 道路边沿掩码 (H, W)
        color: 边沿颜色 (B, G, R)
        thickness: 边沿线粗细
        
    返回:
        可视化结果图像
    """
    result = image.copy()
    
    # 找到边沿点的坐标
    y_coords, x_coords = np.where(edge_mask > 0)
    
    # 在图像上绘制边沿点
    for y, x in zip(y_coords, x_coords):
        cv2.circle(result, (x, y), thickness, color, -1)
    
    return result