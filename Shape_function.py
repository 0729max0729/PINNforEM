import torch
from torch import nn


class ShapeFunction(nn.Module):
    """
    形狀函數：在指定範圍內返回高斯分佈值，範圍外返回 0。
    """
    def __init__(self, radius=3e-2):
        """
        :param radius: 有效範圍的半徑。
        """
        super(ShapeFunction, self).__init__()
        self.radius = radius  # 範圍半徑

    def forward(self, coords):
        """
        前向傳播：計算形狀函數值，範圍外設為0。

        :param coords: (batch_size, n_points, 3) 或相對座標
        :return: (batch_size, n_points) 範圍內的高斯分佈值，範圍外為0。
        """
        # 計算相對座標的歐幾里得距離
        distance = torch.norm(coords, dim=-1)  # (batch_size, n_points)

        # 範圍內使用高斯函數，範圍外設為0
        result = torch.exp(-distance ** 2) * (distance <= self.radius).float()

        return result
