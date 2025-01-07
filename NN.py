import torch
import torch.nn as nn
import torch.nn.functional as F
from pina.model import FeedForward, DeepONet
from pina.model.layers import FourierFeatureEmbedding
from pina.utils import LabelTensor

from Shape_function import ShapeFunction


# 📌 時間子網路
class TimeNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=16):
        super(TimeNet, self).__init__()
        self.fc1 = FeedForward(input_dimensions=input_dim, output_dimensions=output_dim, n_layers=2,inner_size=128)
        self.activation = nn.Tanh()

    def forward(self, t):
        """
        t: LabelTensor (batch_size, 1) 來自 'f' 頻率維度
        """
        t = self.activation(self.fc1(t))
        return t  # (batch_size, output_dim)


# 📌 空間子網路
class SpaceNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=32):
        super(SpaceNet, self).__init__()
        self.fc1 = FeedForward(input_dimensions=input_dim, output_dimensions=hidden_dim, layers=[100])
        self.activation = nn.Tanh()
        self.fourierNet = MultiscaleFourierNet(hidden_dim,hidden_dim)
        self.fc2 = FeedForward(input_dimensions=hidden_dim, output_dimensions=output_dim, n_layers=10,inner_size=128)
    def forward(self, x):
        """
        x: LabelTensor (batch_size, 3) 來自 'x', 'y', 'z' 維度
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fourierNet(x))
        x = self.activation(self.fc2(x))
        return x  # (batch_size, output_dim)


# 📌 融合層
class FusionNet(nn.Module):
    def __init__(self, time_feature_dim=16, space_feature_dim=32, output_dim=32):
        super(FusionNet, self).__init__()
        self.fc1 = nn.Linear(time_feature_dim + space_feature_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)
        self.activation = nn.Tanh()

    def forward(self, time_features, space_features):
        """
        time_features: 時間特徵 (batch_size, time_feature_dim)
        space_features: 空間特徵 (batch_size, space_feature_dim)
        """
        combined = torch.cat((time_features, space_features), dim=1)
        combined = self.activation(self.fc1(combined))
        output = self.activation(self.fc2(combined))
        return output  # (batch_size, output_dim)


# 📌 主網路：時間與空間分離處理
class TimeSpaceNet(nn.Module):
    def __init__(self):
        super(TimeSpaceNet, self).__init__()
        self.time_net = TimeNet(output_dim=8)
        self.space_net = SpaceNet(output_dim=64)
        self.fusion_net = FusionNet(time_feature_dim=8, space_feature_dim=64, output_dim=64)
        self.layers = FeedForward(input_dimensions=64, output_dimensions=2, n_layers=10,inner_size=128)
        self.output_layer = ShapeFunctionModule(ShapeFunction())
    def forward(self, input_tensor: LabelTensor):
        """
        input_tensor: LabelTensor 標記 ['x', 'y', 'z', 'f']
        """
        # 從 LabelTensor 中提取時間和空間維度
        time_input = input_tensor.extract(['f'])/1e6  # 提取時間維度
        space_input = input_tensor.extract(['x', 'y', 'z'])  # 提取空間維度

        # 分別經過時間和空間子網路
        time_features = self.time_net(time_input)
        space_features = self.space_net(space_input)

        # 融合時間與空間特徵
        output = self.fusion_net(time_features, space_features)
        output = self.layers(output)
        output = self.output_layer(output,space_input)
        return LabelTensor(output, labels=['phi_r', 'phi_i'])

class MultiscaleFourierNet(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.embedding1 = FourierFeatureEmbedding(input_dimension=input_dim,
                                                  output_dimension=100,
                                                  sigma=1)
        self.embedding2 = FourierFeatureEmbedding(input_dimension=input_dim,
                                                  output_dimension=100,
                                                  sigma=1)
        self.embedding3 = FourierFeatureEmbedding(input_dimension=input_dim,
                                                  output_dimension=100,
                                                  sigma=1e-2)
        self.final_layers = FeedForward(input_dimensions=300, output_dimensions=output_dim, n_layers=10,inner_size=128)


    def forward(self, x):
        e1 = self.embedding1(x)
        e2 = self.embedding2(x)
        e3 = self.embedding3(x)
        return self.final_layers(torch.cat([e1, e2, e3], dim=-1))


class ShapeFunctionModule(nn.Module):
    """
    使用形狀函數 S(x-x₀, y-y₀, z-z₀) 對 phi 進行加權。
    phi 包含實部 (phi_r) 和虛部 (phi_i)。
    """
    def __init__(self, shape_function):
        """
        :param shape_function: 一個可調用的形狀函數，S(x, y, z)。
        """
        super(ShapeFunctionModule, self).__init__()
        self.shape_function = shape_function  # 傳入自定義的形狀函數

    def forward(self, phi, coords):
        """
        前向傳播：計算形狀函數加權的 phi。

        :param phi: 電勢張量，包含實部與虛部 (batch_size, n_points, 2)
        :param coords: 座標點 (batch_size, n_points, 3)
        :return: 加權後的 phi，包含實部與虛部 (batch_size, n_points, 2)
        """
        # 確保 coords 需要梯度計算
        coords = coords.requires_grad_(True)

        # 計算相對座標 (每個點與所有點的相對距離)
        relative_coords = coords.unsqueeze(1) - coords.unsqueeze(0)  # (batch_size, n_points, n_points, 3)

        # 計算形狀函數 S，並確保其維度一致
        S = self.shape_function(relative_coords)  # (batch_size, n_points, n_points)
        S = S.unsqueeze(-1)  # 增加最後一個維度 (batch_size, n_points, n_points, 1)

        # 對實部和虛部進行加權求和
        phi_sum = phi.unsqueeze(1) * S  # (batch_size, n_points, n_points, 2)
        phi_sum = torch.sum(phi_sum, dim=1)  # (batch_size, n_points, 2)

        return phi_sum


class DEEPONET(nn.Module):
    def __init__(self,input_dim=3,output_dim=2):
        super().__init__()
        branch_net1 = FeedForward(input_dimensions=input_dim, output_dimensions=100)
        trunk_net1 = FeedForward(input_dimensions=1, output_dimensions=100)
        self.deepONet1 = DeepONet(branch_net=branch_net1,
                     trunk_net=trunk_net1,
                     input_indeces_branch_net=['x', 'y', 'z'],
                     input_indeces_trunk_net=['f'],
                     reduction='+',
                     aggregator='*')
        branch_net2 = FeedForward(input_dimensions=input_dim, output_dimensions=100)
        trunk_net2 = FeedForward(input_dimensions=1, output_dimensions=100)
        self.deepONet2 = DeepONet(branch_net=branch_net2,
                     trunk_net=trunk_net2,
                     input_indeces_branch_net=['x', 'y', 'z'],
                     input_indeces_trunk_net=['f'],
                     reduction='+',
                     aggregator='*')

        self.final_layers = FeedForward(input_dimensions=2, output_dimensions=output_dim,
                                        layers=[100, 100])
    def forward(self, x):
        return self.final_layers(torch.cat([self.deepONet1(x),self.deepONet2],dim=-1))





class PointNetSegmentation(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        """
        PointNet 用於逐點特徵提取或分割
        :param input_dim: 每個點的特徵維度 (預設4: x, y, z, f)
        :param output_dim: 每個點的輸出特徵維度
        """
        super(PointNetSegmentation, self).__init__()

        # **1️⃣ 局部特徵提取**
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # **2️⃣ 全局特徵提取**
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # **3️⃣ 特徵融合**
        self.fusion_mlp1 = nn.Sequential(
            nn.Linear(256 + 512, 256),
            nn.ReLU()
        )
        self.fusion_mlp2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # **4️⃣ 輸出層**
        self.output_mlp = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        前向傳播
        :param x: 點雲數據，形狀為 (batch_size, num_points, input_dim)
        :return: 每個點的特徵輸出，形狀為 (batch_size, num_points, output_dim)
        """
        num_points, dimensions = x.shape

        # **Step 1: 局部特徵提取**
        x = x.view(-1, x.shape[-1])  # 展平為 (batch_size * num_points, input_dim)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        # 還原形狀
        x = x.view(1, num_points, -1)  # (batch_size, num_points, 256)
        point_features = x

        # **Step 2: 全局特徵提取**
        global_features = x.permute(0, 2, 1)  # (batch_size, 256, num_points)
        global_features = self.global_max_pool(global_features)  # (batch_size, 256, 1)
        global_features = global_features.reshape([1,-1])  # (batch_size, 256)
        global_features = self.global_mlp(global_features)  # (batch_size, 512)
        global_features = global_features.unsqueeze(1).repeat(1, num_points, 1)  # (batch_size, num_points, 512)

        # **Step 3: 特徵融合**
        fusion_features = torch.cat([point_features, global_features], dim=2)  # (batch_size, num_points, 768)
        fusion_features = self.fusion_mlp1(fusion_features.view(-1, 256 + 512))
        fusion_features = self.fusion_mlp2(fusion_features)

        # 還原形狀
        fusion_features = fusion_features.view(1, num_points, -1)

        # **Step 4: 輸出每個點的特徵**
        output = self.output_mlp(fusion_features)  # (batch_size, num_points, output_dim)

        return output.reshape([num_points,-1])


# 測試網絡
if __name__ == "__main__":
    # 測試模型
    batch_size = 16
    num_points = 1024
    input_dim = 4  # x, y, z, f
    output_dim = 4  # 每個點輸出的特徵維度

    # 模擬點雲數據
    point_cloud = torch.rand(num_points, input_dim)  # 隨機點雲

    # 初始化 PointNet
    model = PointNetSegmentation(input_dim=input_dim, output_dim=output_dim)

    # 前向傳播
    output = model(point_cloud)

    print("輸入形狀:", point_cloud.shape)  # (16, 1024, 4)
    print("輸出形狀:", output.shape)  # (16, 1024, 4)



