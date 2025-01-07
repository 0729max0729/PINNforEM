import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from pina.geometry import Location
from pina.utils import LabelTensor
from sympy.matrices.expressions.blockmatrix import bounds
from scipy.spatial import Voronoi, voronoi_plot_2d


import torch
from pina import LabelTensor


class ConductorLocation(Location):
    """
    Conductor Location: Defines the spatial and frequency sampling for metallic regions.
    """

    def __init__(self, vertices,bound, f_values, sample_mode='interior', z_range=(0, 1), device='cpu'):
        """
        :param vertices: List of (x, y) tuples representing polygon vertices.
        :param f_values: List of frequency values.
        :param sample_mode: Sampling mode ('interior').
        :param z_range: Tuple (z_min, z_max) for Z-axis sampling range.
        :param device: Device for computations ('cpu' or 'cuda').
        """
        super().__init__()
        self.vertices = self._ensure_counter_clockwise(vertices)
        if sample_mode != 'interior':
            raise ValueError("ConductorLocation only supports 'interior' sampling.")
        self.sample_mode = sample_mode
        self.z_range = z_range
        self.f_values = f_values
        self.device = device
        self.bound = bound
    def _ensure_counter_clockwise(self, vertices):
        """
        Ensure the vertices are ordered counter-clockwise.
        """
        area = 0.0
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            area += (x2 - x1) * (y2 + y1)
        if area > 0:
            vertices.reverse()
        return vertices

    def is_inside(self, points):
        """
        Check if points are inside the conductor polygon using ray-casting algorithm.
        """
        result = []
        for point in points:
            x, y = point[:2]
            inside = False
            n = len(self.vertices)
            p1x, p1y = self.vertices[0]
            for i in range(n + 1):
                p2x, p2y = self.vertices[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                p1x, p1y = p2x, p2y
            result.append(inside)
        return torch.tensor(result, dtype=torch.bool, device=self.device)

    def _polygon_area(self):
        """
        計算多邊形的面積 (Shoelace formula)。
        """
        area = 0.0
        n = len(self.vertices)
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2.0

    def _bound_area(self):
        """
        計算多邊形的面積 (Shoelace formula)。
        """
        x = self.bound['x'][1]- self.bound['x'][0]
        y = self.bound['y'][1]- self.bound['y'][0]
        return x*y

    def _sample_interior(self, n):
        """
        Sample points inside the conductor polygon along the Z-axis.
        """
        interior_points = []
        bbox_x = [min(p[0] for p in self.vertices), max(p[0] for p in self.vertices)]
        bbox_y = [min(p[1] for p in self.vertices), max(p[1] for p in self.vertices)]

        while len(interior_points) < n:
            x = torch.rand(1).item() * (bbox_x[1] - bbox_x[0]) + bbox_x[0]
            y = torch.rand(1).item() * (bbox_y[1] - bbox_y[0]) + bbox_y[0]
            z = torch.rand(1).item() * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
            point = torch.tensor([x, y], device=self.device)
            if self.is_inside([point])[0]:
                interior_points.append([x, y, z])

        return interior_points

    def sample(self, n, mode="random", variables=['x', 'y', 'z', 'f']):
        """
        Sample points inside the conductor region, with edge attraction effect,
        including XY plane edges and Z-axis boundaries, and combine with frequency values.
        """
        #n=n*self._polygon_area()/self._bound_area()
        edge_ratio = 0.3
        # **1️⃣ 原始點樣本**
        sampled_points = self._sample_interior(int(n * (1 - edge_ratio)))
        f_values = torch.tensor(self.f_values, device=self.device)

        sampled_points_with_f = [
            (point[0], point[1], point[2], f.item()) for point in sampled_points for f in f_values
        ]

        for _ in range(int(n * edge_ratio)):
            # **2️⃣ 額外生成邊緣點**
            edge_points = []
            additional_points = int(n * edge_ratio)

            edges = [(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)])
                     for i in range(len(self.vertices))]
            edge = random.choice(edges)

            # 計算投影點 (邊緣上隨機插值)
            (x1, y1), (x2, y2) = edge
            z_min, z_max = self.z_range
            t = torch.rand(1).item()  # 線性插值比例
            x = x1 * (1 - t) + x2 * t
            y = y1 * (1 - t) + y2 * t
            z = torch.rand(1).item() * (z_max - z_min) + z_min

            edge_points.append((x, y, z))

        # 加入頻率維度
        edge_points_with_f = [
            (point[0], point[1], point[2], f.item()) for point in edge_points for f in f_values
        ]

        # **3️⃣ 合併原始點和邊緣點**
        all_points_with_f = sampled_points_with_f + edge_points_with_f

        return LabelTensor(torch.tensor(all_points_with_f, dtype=torch.float32, device=self.device),
                           labels=['x', 'y', 'z', 'f'])


class DielectricLocation(Location):
    """
    Dielectric Location: Defines spatial and frequency sampling for dielectric regions,
    excluding conductor regions.
    """

    def __init__(self, conductors, bound, f_values, sample_mode='outer', z_range=(0, 1), device='cpu'):
        """
        :param conductors: List of ConductorLocation objects (to exclude from sampling).
        :param bound: Dictionary defining domain boundaries.
        :param f_values: List of frequency values.
        :param sample_mode: Sampling mode ('outer').
        :param z_range: Tuple (z_min, z_max) for Z-axis sampling range.
        :param device: Device for computations ('cpu' or 'cuda').
        """
        super().__init__()
        if sample_mode != 'outer':
            raise ValueError("DielectricLocation only supports 'outer' sampling.")
        self.conductors = conductors
        self.z_range = z_range
        self.f_values = f_values
        self.device = device
        self.bound = bound

    def is_inside(self, point):
        """
        Check if a point is inside any conductor region.
        """
        for conductor in self.conductors:
            if conductor.is_inside([point])[0]:
                return True
        return False

    def _sample_outer(self, n):
        """
        Sample points in the dielectric region, excluding conductor regions.
        """
        outer_points = []
        bbox_x = [self.bound['x'][0], self.bound['x'][1]]
        bbox_y = [self.bound['y'][0], self.bound['y'][1]]

        while len(outer_points) < n:
            x = torch.rand(1).item() * (bbox_x[1] - bbox_x[0]) + bbox_x[0]
            y = torch.rand(1).item() * (bbox_y[1] - bbox_y[0]) + bbox_y[0]
            z = torch.rand(1).item() * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
            point = torch.tensor([x, y], device=self.device)
            if not self.is_inside(point):
                outer_points.append([x, y, z])

        return outer_points



    def sample(self, n,mode="random", variables=['x', 'y', 'z', 'f']):
        """
        Sample points in the dielectric region, 在原始點基礎上增加邊緣點。

        :param n: 總點數。
        :param edge_ratio: 邊緣點比例 (0.0 - 1.0)。
        :param variables: 樣本變量名稱。
        :return: LabelTensor。
        """
        edge_ratio=0.3
        # **1️⃣ 原始點樣本**
        sampled_points = self._sample_outer(n)
        f_values = torch.tensor(self.f_values, device=self.device)

        sampled_points_with_f = [
            (point[0], point[1], point[2], f.item()) for point in sampled_points for f in f_values
        ]
        '''
        for conductor in self.conductors:
            for _ in range(int(n * edge_ratio)):
                # **2️⃣ 額外生成邊緣點**
                edge_points = []
                additional_points = int(n * edge_ratio)

                edges = [(conductor.vertices[i], conductor.vertices[(i + 1) % len(conductor.vertices)])
                         for i in range(len(conductor.vertices))]
                edge = random.choice(edges)

                # 計算投影點 (邊緣上隨機插值)
                (x1, y1), (x2, y2) = edge
                z_min, z_max = self.z_range
                t = torch.rand(1).item()  # 線性插值比例
                x = x1 * (1 - t) + x2 * t
                y = y1 * (1 - t) + y2 * t
                z = torch.rand(1).item() * (z_max - z_min) + z_min

                edge_points.append((x, y, z))

        # 加入頻率維度
        edge_points_with_f = [
            (point[0], point[1], point[2], f.item()) for point in edge_points for f in f_values
        ]

        # **3️⃣ 合併原始點和邊緣點**
        all_points_with_f = sampled_points_with_f + edge_points_with_f
        '''
        return LabelTensor(torch.tensor(sampled_points_with_f, dtype=torch.float32, device=self.device),
                           labels=['x', 'y', 'z', 'f'])


import torch
from pina.label_tensor import LabelTensor


class PortLocation(Location):
    """
    定義 PortLocation，用於指定初始條件的位置。
    - 空間坐標: (x, y, z) 為單點。
    - 頻率範圍: 可以是一個頻率範圍 (f_min, f_max) 或頻率列表 [f1, f2, ...]。
    """

    def __init__(self, point, frequencies, device='cpu'):
        """
        :param point: 單點 (x, y, z) 作為空間位置。
        :param frequencies: 頻率列表或範圍 (list of float)。
        :param device: 使用裝置 (CPU 或 CUDA)。
        """
        super().__init__()
        if len(point) != 3:
            raise ValueError("point 必須是 (x, y, z) 的單點坐標。")

        self.point = point  # (x, y, z)
        self.frequencies = frequencies if isinstance(frequencies, list) else [frequencies]
        self.device = device

    def is_inside(self, points):
        pass

    def sample(self, n, mode='all', variables=['x', 'y', 'z', 'f']):
        """
        生成樣本點，包含 (x, y, z) 和頻率範圍內的所有頻率，並使用 torch.repeat 增加密度。

        :param n: 每個頻率下生成的樣本點數量。
        :param mode: 'all' 表示返回所有頻率的樣本點。
        :param variables: 返回的樣本點包含的變量名稱。
        :return: LabelTensor 包含 x, y, z, f。
        """
        # **1️⃣ 創建空間坐標 (x, y, z)**
        point = torch.tensor(self.point, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 3]
        points = point.repeat(len(self.frequencies) * n, 1)  # 重複點數 [len(freq) * n, 3]

        # **2️⃣ 創建頻率維度**
        freq = torch.tensor(self.frequencies, dtype=torch.float32, device=self.device).unsqueeze(1)  # [len(freq), 1]
        freq = freq.repeat_interleave(n, dim=0)  # [len(freq) * n, 1]

        # **3️⃣ 合併成 (x, y, z, f)**
        samples = torch.cat([points, freq], dim=1)  # [len(freq) * n, 4]

        # **4️⃣ 返回 LabelTensor**
        return LabelTensor(samples, labels=variables)


# 測試 3D PolygonLocation

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from pina import LabelTensor

    # 定義兩個金屬塊的頂點
    vertices_metal1 = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0)
    ]

    vertices_metal2 = [
        (1.5, 0.0),
        (2.5, 0.0),
        (1.5, 1.0)
    ]

    # 定義介質的邊界範圍
    bound_dielectric = {
        'x': [-1.0, 3.0],
        'y': [-1.0, 2.0],
        'z': [0.0, 1.0]
    }

    # 頻率範圍
    f_values = [1e9]

    # 定義金屬塊位置
    metal1 = ConductorLocation(
        vertices=vertices_metal1,
        f_values=f_values,
        sample_mode='interior',
        z_range=(0, 1),
        device='cpu'
    )

    metal2 = ConductorLocation(
        vertices=vertices_metal2,
        f_values=f_values,
        sample_mode='interior',
        z_range=(0, 1),
        device='cpu'
    )

    # 定義介質區域
    dielectric = DielectricLocation(
        conductors=[metal1, metal2],
        bound=bound_dielectric,
        f_values=f_values,
        sample_mode='outer',
        z_range=(0, 1),
        device='cpu'
    )

    # 每個金屬塊採樣 1000 個點
    samples_metal1 = metal1.sample(n=1000)
    samples_metal2 = metal2.sample(n=1000)

    # 介質區域採樣 2000 個點
    samples_dielectric = dielectric.sample(n=2000)

    # 轉換為 numpy 進行繪圖
    points_metal1 = samples_metal1[:, :3].cpu().numpy()
    points_metal2 = samples_metal2[:, :3].cpu().numpy()
    points_dielectric = samples_dielectric[:, :3].cpu().numpy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 繪製金屬塊 1
    ax.scatter(
        points_metal1[:, 0], points_metal1[:, 1], points_metal1[:, 2],
        c='red', marker='o', s=1, label='Metal Block 1'
    )

    # 繪製金屬塊 2
    ax.scatter(
        points_metal2[:, 0], points_metal2[:, 1], points_metal2[:, 2],
        c='blue', marker='o', s=1, label='Metal Block 2'
    )

    # 繪製介質區域
    ax.scatter(
        points_dielectric[:, 0], points_dielectric[:, 1], points_dielectric[:, 2],
        c='green', marker='o', s=1, label='Dielectric'
    )

    # 添加標籤和標題
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Sampling Points for Two Metal Blocks and Dielectric')
    ax.legend()

    plt.show()

