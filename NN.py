import torch
import torch.nn as nn
from pina.model.layers import FourierFeatureEmbedding


class MultiscaleFourierNet(nn.Module):
    def __init__(self, input_dimension=4, output_dimension=6):
        """
        多尺度傅里葉神經網絡，用於 3D Maxwell 方程的數值模擬。

        :param input_dimension: 輸入維度 (例如: x, y, z, t => 4 維)
        :param output_dimension: 輸出維度 (例如: E_x, E_y, E_z, H_x, H_y, H_z => 6 維)
        """
        super(MultiscaleFourierNet, self).__init__()

        # 多尺度傅里葉嵌入
        self.embedding1 = FourierFeatureEmbedding(
            input_dimension=input_dimension,
            output_dimension=64,
            sigma=1
        )
        self.embedding2 = FourierFeatureEmbedding(
            input_dimension=input_dimension,
            output_dimension=256,
            sigma=1e3
        )
        self.embedding3 = FourierFeatureEmbedding(
            input_dimension=input_dimension,
            output_dimension=256,
            sigma=1e4
        )
        self.embedding4 = FourierFeatureEmbedding(
            input_dimension=input_dimension,
            output_dimension=256,
            sigma=1e5
        )
        # 前饋神經網絡
        self.layers = nn.Sequential(

            nn.Linear(64, 128),
            nn.Linear(128, output_dimension)  # 輸出映射到物理變量數量
        )

    def forward(self, x):
        """
        前向傳播
        :param x: 輸入點 (torch.Tensor), 形狀: [batch_size, input_dimension]
        :return: 輸出 (torch.Tensor), 形狀: [batch_size, output_dimension]
        """
        # 低頻和高頻嵌入
        e1 = self.embedding1(x)  # 低頻嵌入
        e2 = self.embedding2(x)  # 高频嵌入
        e3 = self.embedding3(x)  # 高频嵌入
        e4 = self.embedding4(x)  # 高频嵌入
        # 拼接特徵
        combined = torch.cat([e1], dim=-1)

        # 通過神經網絡
        output = self.layers(combined)
        return output


import torch
from pina.utils import LabelTensor


class SinCosFeature3D(torch.nn.Module):
    """
    3D Maxwell 額外特徵: A * sin(alpha*x + phi_x) * sin(beta*y + phi_y) * sin(gamma*z + phi_z) * cos(delta*t + phi_t)
    """

    def __init__(self):
        super().__init__()
        # 頻率參數
        self.alpha = torch.nn.Parameter(torch.tensor([1.0]))
        self.beta = torch.nn.Parameter(torch.tensor([1.0]))
        self.gamma = torch.nn.Parameter(torch.tensor([1.0]))
        self.delta = torch.nn.Parameter(torch.tensor([1.0]))

        # 相位參數
        self.phi_x = torch.nn.Parameter(torch.tensor([0.0]))
        self.phi_y = torch.nn.Parameter(torch.tensor([0.0]))
        self.phi_z = torch.nn.Parameter(torch.tensor([0.0]))
        self.phi_t = torch.nn.Parameter(torch.tensor([0.0]))

        # 振幅參數
        self.amplitude = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        """
        x: LabelTensor 包含 ['x', 'y', 'z', 't']
        """
        x_coord = x.extract(['x'])
        y_coord = x.extract(['y'])
        z_coord = x.extract(['z'])

        feature = (
                self.amplitude *
                torch.sin(self.alpha * x_coord + self.phi_x) *
                torch.sin(self.beta * y_coord + self.phi_y) *
                torch.sin(self.gamma * z_coord + self.phi_z)
        )
        return LabelTensor(feature, ['A*sin(a*x+phi_x)sin(b*y+phi_y)sin(c*z+phi_z)cos(d*t+phi_t)'])


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierLayer3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        """
        modes1: x 方向傅里叶模式数量
        modes2: y 方向傅里叶模式数量
        modes3: z 方向傅里叶模式数量 (在 rfft 中是 N/2+1)
        width: 网络宽度（通道数）
        """
        super(FourierLayer3D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width

        self.scale = 1 / (width * width)
        self.weights = nn.Parameter(
            self.scale * torch.rand(width, width, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        actual_modes1 = min(self.modes1, x_ft.shape[-3])
        actual_modes2 = min(self.modes2, x_ft.shape[-2])
        actual_modes3 = min(self.modes3, x_ft.shape[-1])

        out_ft = torch.zeros_like(x_ft)
        for i in range(self.width):
            out_ft[:, i, :actual_modes1, :actual_modes2, :actual_modes3] = (
                    x_ft[:, i, :actual_modes1, :actual_modes2, :actual_modes3] *
                    self.weights[i, i, :actual_modes1, :actual_modes2, :actual_modes3]
            )

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, input_dim, output_dim, grid_size):
        """
        modes1, modes2, modes3: 每个方向上的傅里叶模式数
        width: 每层的隐藏通道数
        input_dim: 输入特征维度 (4: x, y, z, f)
        output_dim: 输出特征维度 (2)
        grid_size: (grid_x, grid_y, grid_z) 目标网格大小
        """
        super(FNO3D, self).__init__()
        self.width = width
        self.grid_size = grid_size

        # Lifting Layer: 将输入映射到 3D 空间
        self.fc0 = nn.Linear(input_dim, width * grid_size[0] * grid_size[1] * grid_size[2])

        # Fourier Layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer3D(modes1, modes2, modes3, self.width) for _ in range(4)
        ])

        # 非线性激活函数
        self.activation = nn.GELU()

        # Projection Layers
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, output_dim)

    def forward(self, x):
        """
        x: (batch, 4)
        """
        batch_size = x.shape[0]

        # Lifting Layer: 映射到 3D 网格
        x = self.fc0(x)  # (batch, grid_x * grid_y * grid_z * width)
        x = x.view(batch_size, self.width, *self.grid_size)  # (batch, channels, grid_x, grid_y, grid_z)
        x = self.activation(x)

        # Fourier Layers
        for layer in self.fourier_layers:
            x = layer(x)
            x = self.activation(x)

        # 全局平均池化
        x = torch.mean(x, dim=(-3, -2, -1))  # (batch, channels)

        # Linear projection to output_dim
        x = self.fc1(x)  # (batch, width)
        x = self.activation(x)
        x = self.fc2(x)  # (batch, output_dim)

        return x


# 測試網絡
if __name__ == "__main__":
    # 模拟输入
    batch_size = 4
    input_dim = 4  # 输入特征维度
    output_dim = 2  # 输出特征维度
    grid_size = (16, 16, 16)  # 目标网格大小

    # 输入维度: (batch_size, input_dim)
    x = torch.rand(batch_size, input_dim).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义 FNO3D 模型
    fno3d_model = FNO3D(
        modes1=12,
        modes2=12,
        modes3=12,
        width=32,
        input_dim=input_dim,
        output_dim=output_dim,
        grid_size=grid_size
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 前向传播
    output = fno3d_model(x)

    # 输出形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


