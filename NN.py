import torch
import torch.nn as nn
import torch.nn.functional as F
from pina.utils import LabelTensor


# 📌 時間子網路
class TimeNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=16):
        super(TimeNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, t):
        """
        t: LabelTensor (batch_size, 1) 來自 'f' 頻率維度
        """
        t = self.activation(self.fc1(t))
        t = self.activation(self.fc2(t))
        return t  # (batch_size, output_dim)


# 📌 空間子網路
class SpaceNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=32):
        super(SpaceNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: LabelTensor (batch_size, 3) 來自 'x', 'y', 'z' 維度
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x  # (batch_size, output_dim)


# 📌 融合層
class FusionNet(nn.Module):
    def __init__(self, time_feature_dim=16, space_feature_dim=32, output_dim=2):
        super(FusionNet, self).__init__()
        self.fc1 = nn.Linear(time_feature_dim + space_feature_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.activation = nn.ReLU()

    def forward(self, time_features, space_features):
        """
        time_features: 時間特徵 (batch_size, time_feature_dim)
        space_features: 空間特徵 (batch_size, space_feature_dim)
        """
        combined = torch.cat((time_features, space_features), dim=1)
        combined = self.activation(self.fc1(combined))
        output = self.fc2(combined)
        return output  # (batch_size, output_dim)


# 📌 主網路：時間與空間分離處理
class TimeSpaceNet(nn.Module):
    def __init__(self):
        super(TimeSpaceNet, self).__init__()
        self.time_net = TimeNet(output_dim=1)
        self.space_net = SpaceNet(output_dim=32)
        self.fusion_net = FusionNet(time_feature_dim=1, space_feature_dim=32)

    def forward(self, input_tensor: LabelTensor):
        """
        input_tensor: LabelTensor 標記 ['x', 'y', 'z', 'f']
        """
        # 從 LabelTensor 中提取時間和空間維度
        time_input = input_tensor.extract(['f'])/1e8  # 提取時間維度
        space_input = input_tensor.extract(['x', 'y', 'z'])*1e7  # 提取空間維度

        # 分別經過時間和空間子網路
        time_features = self.time_net(time_input)
        space_features = self.space_net(space_input)

        # 融合時間與空間特徵
        output = self.fusion_net(time_features, space_features)
        return LabelTensor(output, labels=['phi_r', 'phi_i'])



# 測試網絡
if __name__ == "__main__":
    # 模拟输入
    batch_size = 4
    input_dim = 4  # 输入特征维度
    output_dim = 2  # 输出特征维度
    grid_size = (32, 32, 32)  # 目标网格大小

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


