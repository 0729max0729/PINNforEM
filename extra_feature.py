import torch
import torch.nn as nn
from pina.label_tensor import LabelTensor


class HelmholtzFeature(nn.Module):
    """
    基於 Helmholtz 方程齊性解的 Extra Feature 模型，包含可學習的波數 (k)、幅度 (A) 和波源位置 (x0, y0, z0)。
    """

    def __init__(self, num_sources=3):
        """
        初始化可學習參數 A、k 和波源位置。
        :param num_sources: 波源的數量。
        """
        super().__init__()
        self.num_sources = num_sources

        # 可學習的幅度和波數
        self.A = nn.Parameter(torch.ones(num_sources, dtype=torch.float32))  # 每個波源的幅度
        self.k = nn.Parameter(torch.ones(num_sources, dtype=torch.float32))  # 每個波源的波數

        # 可學習的波源位置
        self.source_positions = nn.Parameter(torch.rand(num_sources, 3, dtype=torch.float32))  # [x0, y0, z0]

    def forward(self, x):
        """
        計算齊性解特徵，考慮多個波源。
        :param x: LabelTensor, 包含 ['x', 'y', 'z']
        :return: LabelTensor, 包含 Helmholtz 方程的齊性解特徵。
        """
        x_coord = x.extract(['x'])  # (batch_size, 1)
        y_coord = x.extract(['y'])  # (batch_size, 1)
        z_coord = x.extract(['z'])  # (batch_size, 1)

        phi_total = torch.zeros_like(x_coord)  # 初始化總場

        # 對每個波源進行場型疊加
        for i in range(self.num_sources):
            x0, y0, z0 = self.source_positions[i]
            r = torch.sqrt((x_coord - x0) ** 2 + (y_coord - y0) ** 2 + (z_coord - z0) ** 2 + 1e-8)  # 避免除以零

            # 計算每個波源的場分布（球面波為例）
            phi = self.A[i] * torch.sin(self.k[i] * r) / r  # 球面波解
            phi_total += phi

        return LabelTensor(phi_total, labels=['HelmholtzFeature'])
