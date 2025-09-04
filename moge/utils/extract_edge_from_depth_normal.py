import numpy as np
import cv2

def extract_edge_from_depth_normal(
    depth: np.ndarray,
    normal: np.ndarray = None,
    ksize: int = 3,
    mode: str = "raw"
) -> np.ndarray:
    """
    从深度图和法线图中提取边缘图 (edge_map)。

    参数:
        depth (np.ndarray): 深度图, shape [H, W] 或 [B, H, W]。
        normal (np.ndarray): 法线图, shape [H, W, 3] 或 [B, H, W, 3]，可选。
        ksize (int): Sobel算子核大小 (默认 3)。
        mode (str): 边缘后处理方式:
            - "raw"    : 原始细节丰富的边缘
            - "thresh" : 阈值化，保留强边缘
            - "canny"  : Canny边缘检测，突出主要线条
            - "morph"  : 形态学处理，去噪并加粗边缘

    返回:
        edge_map (np.ndarray): 边缘图，范围 [0,1]，shape [H,W]。
    """
    def _normalize(x, eps=1e-6):
        x = np.nan_to_num(x, nan=0.0)
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    # 处理输入
    depth_map = np.nan_to_num(depth, nan=0.0).astype(np.float32)
    normal_map = None
    if normal is not None:
        normal_map = np.nan_to_num(normal, nan=0.0).astype(np.float32)

    # 深度边缘
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=ksize)
    dx, dy = np.clip(dx, -1e6, 1e6), np.clip(dy, -1e6, 1e6)
    depth_edge = np.sqrt(dx ** 2 + dy ** 2)
    depth_edge = _normalize(depth_edge)

    # 法线边缘
    normal_edge = 0
    if normal_map is not None:
        nx = cv2.Sobel(normal_map[..., 0], cv2.CV_32F, 1, 0, ksize=ksize) + \
             cv2.Sobel(normal_map[..., 0], cv2.CV_32F, 0, 1, ksize=ksize)
        ny = cv2.Sobel(normal_map[..., 1], cv2.CV_32F, 1, 0, ksize=ksize) + \
             cv2.Sobel(normal_map[..., 1], cv2.CV_32F, 0, 1, ksize=ksize)
        nz = cv2.Sobel(normal_map[..., 2], cv2.CV_32F, 1, 0, ksize=ksize) + \
             cv2.Sobel(normal_map[..., 2], cv2.CV_32F, 0, 1, ksize=ksize)
        normal_edge = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        normal_edge = _normalize(normal_edge)

    # 融合
    edge = _normalize(normal_edge)
    # depth_edge + normal_edge

    # ===== 后处理 =====
    if mode == "thresh":
        edge = (edge > 0.1).astype(np.float32)

    elif mode == "canny":
        edge_blur = cv2.GaussianBlur(edge, (5, 5), 1.0)
        edge_canny = cv2.Canny((edge_blur * 255).astype(np.uint8), 30, 100)
        edge = edge_canny.astype(np.float32) / 255.0

    elif mode == "morph":
        edge_binary = (edge > 0.1).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        edge_clean = cv2.morphologyEx(edge_binary, cv2.MORPH_OPEN, kernel)
        edge_main = cv2.dilate(edge_clean, kernel, iterations=1)
        edge = edge_main.astype(np.float32)

    # "raw" 直接返回原始融合结果
    return edge