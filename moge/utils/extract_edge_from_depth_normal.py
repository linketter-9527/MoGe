import numpy as np
import cv2

def extract_edge_from_depth_normal(depth: np.ndarray, normal: np.ndarray = None, ksize: int = 3) -> np.ndarray:
    """
    从深度图和法线图中提取边缘图 (edge_map)。

    参数:
        depth (np.ndarray): 深度图, shape [H, W] 或 [B, H, W]。
        normal (np.ndarray): 法线图, shape [H, W, 3] 或 [B, H, W, 3]，可选。
        ksize (int): Sobel算子核大小 (默认 3)。

    返回:
        edge_map (np.ndarray): 边缘图，范围 [0,1]，shape [H,W] 或 [B,H,W]。
    """
    def _normalize(x, eps=1e-6):
        x = x - x.min()
        return x / (x.max() + eps)

    def _compute_single_edge(depth_map, normal_map=None):
        # 深度边缘：Sobel
        depth_map = depth_map.astype(np.float32)
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=ksize)
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=ksize)
        depth_edge = np.sqrt(dx ** 2 + dy ** 2)
        depth_edge = _normalize(depth_edge)

        # 法线边缘
        normal_edge = 0
        if normal_map is not None:
            normal_map = normal_map.astype(np.float32)
            # 计算相邻像素法线余弦相似度
            nx = cv2.Sobel(normal_map[..., 0], cv2.CV_32F, 1, 0, ksize=ksize) + \
                 cv2.Sobel(normal_map[..., 0], cv2.CV_32F, 0, 1, ksize=ksize)
            ny = cv2.Sobel(normal_map[..., 1], cv2.CV_32F, 1, 0, ksize=ksize) + \
                 cv2.Sobel(normal_map[..., 1], cv2.CV_32F, 0, 1, ksize=ksize)
            nz = cv2.Sobel(normal_map[..., 2], cv2.CV_32F, 1, 0, ksize=ksize) + \
                 cv2.Sobel(normal_map[..., 2], cv2.CV_32F, 0, 1, ksize=ksize)
            normal_edge = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
            normal_edge = _normalize(normal_edge)

        # 融合
        edge = depth_edge + normal_edge
        return _normalize(edge)

    # 处理 batch
    if depth.ndim == 2:  # 单张
        return _compute_single_edge(depth, normal)
    elif depth.ndim == 3:  # batch [B,H,W]
        edges = []
        for i in range(depth.shape[0]):
            nmap = normal[i] if normal is not None else None
            edges.append(_compute_single_edge(depth[i], nmap))
        return np.stack(edges, axis=0)
    else:
        raise ValueError("depth 输入必须是 [H,W] 或 [B,H,W]")