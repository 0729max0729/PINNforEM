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



# 測試網絡
if __name__ == "__main__":
    model = MultiscaleFourierNet(input_dimension=4, output_dimension=6)
    x = torch.rand(10, 4)  # 10個樣本，4維輸入 (x, y, z, t)
    output = model(x)
    print("輸入形狀:", x.shape)  # torch.Size([10, 4])
    print("輸出形狀:", output.shape)  # torch.Size([10, 6])
