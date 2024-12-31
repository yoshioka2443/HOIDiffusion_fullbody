import numpy as np
import torch

def apply_transform(vertices, transform_matrix):
    """頂点に4x4の変換行列を適用します。"""
    transformed_vertices = vertices @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]
    return transformed_vertices.to(torch.float32)

def to_homogeneous_matrix(R=None, T=None):
    """回転行列と並進ベクトルから4x4の同次変換行列を作成します。"""
    transform_matrix = np.eye(4)
    if R is not None:
        transform_matrix[:3, :3] = R
    if T is not None:
        transform_matrix[:3, 3] = T
    return transform_matrix