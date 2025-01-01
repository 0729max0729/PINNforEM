import numpy as np
import torch
from pina.geometry import Location
from pina.utils import LabelTensor
from sympy.matrices.expressions.blockmatrix import bounds
from scipy.spatial import Voronoi, voronoi_plot_2d


class PolygonLocation(Location):
    """
    Implementation of a 3D Polygon Domain with flexible sampling based on a sample_mode label,
    including normal vectors in both XY plane and Z direction, and frequency domain support.
    """

    def __init__(self, vertices, bound, f_values, sample_mode='interior', z_range=(0, 1), device='cpu',scale=1e-6):
        """
        :param vertices: List of (x, y) tuples representing polygon vertices (base).
        :param sample_mode: Sampling mode ('interior', 'edges', 'both').
        :param z_range: Tuple (z_min, z_max) representing the Z-axis range.
        :param freq_range: Tuple (f_min, f_max) representing the frequency range for sampling.
        :param device: Device for tensor computations ('cpu' or 'cuda').
        """
        super().__init__()

        if sample_mode not in ['interior', 'edges', 'outer']:
            raise ValueError("sample_mode must be 'interior', 'edges', or 'both'")
        self.sample_mode = sample_mode
        self.z_range = z_range  # Z 軸範圍
        self.f_values = f_values  # 頻率範圍
        self.device = device
        self.bound = bound
        self.scale = scale
        self.vertices = self._ensure_counter_clockwise(vertices)
    def _ensure_counter_clockwise(self, vertices):
        """
        Ensure the vertices are ordered counter-clockwise.
        """
        area = 0.0
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            area += (x2/self.scale - x1/self.scale) * (y2/self.scale + y1/self.scale)

        if area > 0:  # If area is positive, the vertices are clockwise
            vertices.reverse()
        return vertices

    def is_inside(self, points):
        """
        Check if points are inside the polygon or on its edges using ray-casting algorithm.
        If the polygon is defined by edges only, checks if points lie on the edges.
        """
        result = []
        epsilon = 1e-6  # 浮點數誤差閾值

        for point in points:
            x, y = point[:2]  # 只考慮 X, Y 平面
            inside = False
            on_edge = False  # 新增邊界標誌

            n = len(self.vertices)
            p1x, p1y = self.vertices[0]

            for i in range(n + 1):
                p2x, p2y = self.vertices[i % n]

                # **1️⃣ 檢查是否在邊緣上**
                if (min(p1x, p2x) - epsilon <= x <= max(p1x, p2x) + epsilon and
                        min(p1y, p2y) - epsilon <= y <= max(p1y, p2y) + epsilon):
                    # 計算線段與點的最小距離
                    edge_length = ((p2x - p1x) ** 2 + (p2y - p1y) ** 2) ** 0.5
                    if edge_length > epsilon:  # 避免除以零
                        distance = abs((p2y - p1y) * x - (p2x - p1x) * y + p2x * p1y - p2y * p1x) / edge_length
                        if distance < epsilon:
                            on_edge = True

                # **2️⃣ 判斷內部 (射線法)**
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside

                p1x, p1y = p2x, p2y

            # **3️⃣ 結合內部和邊界判斷**
            if inside or on_edge:
                result.append(True)
            else:
                result.append(False)

        return torch.tensor(result, dtype=torch.bool, device=self.device)

    def _sample_on_edges(self, n):
        """
        Sample points uniformly on the surface of a polygonal prism,
        including top and bottom faces and side walls.
        """
        edge_points = []

        edges_xy = [(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]) for i in range(len(self.vertices))]
        num_points_per_section = n // 3  # 分為底面、頂面、側面

        total_length = sum(torch.sqrt((torch.tensor(p2[0]) - torch.tensor(p1[0])) ** 2 +
                                      (torch.tensor(p2[1]) - torch.tensor(p1[1])) ** 2)
                           for p1, p2 in edges_xy)

        points_per_edge = [int(n * torch.sqrt((torch.tensor(p2[0]) - torch.tensor(p1[0])) ** 2 +
                                              (torch.tensor(p2[1]) - torch.tensor(p1[1])) ** 2) / total_length)
                           for p1, p2 in edges_xy]

        for (p1, p2), num_points in zip(edges_xy, points_per_edge):
            for i in range(num_points):
                t = i / max(num_points - 1, 1)
                x = p1[0] * (1 - t) + p2[0] * t
                y = p1[1] * (1 - t) + p2[1] * t
                z = torch.rand(1).item() * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
                edge_points.append([x, y, z])

        ## 1️⃣ **在 XY 平面 (z_min 和 z_max) 的內部均勻取樣**
        for z in [self.z_range[0], self.z_range[1]]:
            bbox_x = [min(p[0] for p in self.vertices), max(p[0] for p in self.vertices)]
            bbox_y = [min(p[1] for p in self.vertices), max(p[1] for p in self.vertices)]

            sampled_points_xy = 0
            attempts = 0
            max_attempts = 99999  # 最大嘗試次數，防止死循環
            while sampled_points_xy < num_points_per_section and attempts < max_attempts:
                x = torch.rand(1).item() * (bbox_x[1] - bbox_x[0]) + bbox_x[0]
                y = torch.rand(1).item() * (bbox_y[1] - bbox_y[0]) + bbox_y[0]
                point = torch.tensor([x, y])
                if self.is_inside([point])[0]:
                    edge_points.append([x, y, z])
                    sampled_points_xy += 1

                attempts += 1

        return edge_points

    def _sample_interior(self, n):
        """
        Sample points inside the polygon, extended into the Z-axis.
        """
        interior_points = []
        bbox_x = [min(p[0] for p in self.vertices), max(p[0] for p in self.vertices)]
        bbox_y = [min(p[1] for p in self.vertices), max(p[1] for p in self.vertices)]

        while len(interior_points) < n:
            x = torch.rand(1) * (bbox_x[1] - bbox_x[0]) + bbox_x[0]
            y = torch.rand(1) * (bbox_y[1] - bbox_y[0]) + bbox_y[0]
            z = torch.rand(1).item() * (self.z_range[1]/self.scale - self.z_range[0]/self.scale) + self.z_range[0]/self.scale
            point = torch.tensor([x.item(), y.item()], device=self.device)
            if self.is_inside([point])[0]:
                interior_points.append([x.item(), y.item(), z])

        return interior_points

    def _sample_outer(self, n):
        """
        Sample points inside the polygon, extended into the Z-axis.
        """
        interior_points = []
        bbox_x = [self.bound['x'][0]/self.scale, self.bound['x'][1]/self.scale]
        bbox_y = [self.bound['y'][0]/self.scale, self.bound['y'][1]/self.scale]

        while len(interior_points) < n:
            x = torch.rand(1) * (bbox_x[1] - bbox_x[0]) + bbox_x[0]
            y = torch.rand(1) * (bbox_y[1] - bbox_y[0]) + bbox_y[0]
            z = torch.rand(1).item() * (self.z_range[1]/self.scale - self.z_range[0]/self.scale) + self.z_range[0]/self.scale
            point = torch.tensor([x.item(), y.item()], device=self.device)
            if not(self.is_inside([point])[0]):
                interior_points.append([x.item(), y.item(), z])

        return interior_points

    def calculate_normal_vector(self):
        """
        Automatically calculate the normal vector for edge segments in 3D space.
        """
        normal_vectors = []
        edges = [(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]) for i in range(len(self.vertices))]
        z_min, z_max = self.z_range
        if z_min < z_max:
            # 計算 XY 平面上的邊法向量
            for (p1, p2) in edges:
                # 定義兩個邊緣點 (p1, p2) 和 z 軸上下界，構建三個點來定義一個平面
                point1 = torch.tensor([p1[0], p1[1], z_min], dtype=torch.float32, device=self.device)
                point2 = torch.tensor([p2[0], p2[1], z_min], dtype=torch.float32, device=self.device)
                point3 = torch.tensor([p1[0], p1[1], z_max], dtype=torch.float32, device=self.device)

                # 計算兩個邊緣向量
                edge_vector_1 = point2 - point1  # 邊的方向向量
                edge_vector_2 = point3 - point1  # 垂直於邊的 z 軸方向向量

                # 計算叉積以獲得法向量
                normal = torch.cross(edge_vector_1, edge_vector_2, dim=0)
                if not (all(normal == 0)):
                    normal = normal / torch.norm(normal)  # 正規化為單位向量

                    normal_vectors.append(normal)
        if len(list(set(self.vertices))) > 2:
            # 添加平行於 Z 軸的法向量（上表面和下表面）
            top_surface_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)  # Z 軸正方向
            bottom_surface_normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)  # Z 軸負方向

            # 將上表面和下表面的法向量分別添加到法向量列表中

            normal_vectors.append(top_surface_normal)
            normal_vectors.append(bottom_surface_normal)

        return normal_vectors

    def sample(self, n, mode='random', variables=['x', 'y', 'z', 'f']):
        """
        Sample points in 3D space, including Z and frequency dimensions.
        """

        if self.sample_mode == 'interior':
            sampled_points = self._sample_interior(n)
        elif self.sample_mode == 'edges':
            sampled_points = self._sample_on_edges(n)
        elif self.sample_mode == 'outer':
            sampled_points = self._sample_outer(n)

        # 添加頻率維度取樣
        f_values = torch.tensor(self.f_values, device=self.device)

        # 合併 x, y, z, f
        sampled_points_with_f = [
            (point[0]*self.scale, point[1]*self.scale, point[2]*self.scale, f.item()) for point in sampled_points for f in f_values
        ]

        return LabelTensor(torch.tensor(sampled_points_with_f, dtype=torch.float32, device=self.device),
                           labels=['x', 'y', 'z', 'f'])


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
        生成樣本點，包含 (x, y, z) 和頻率範圍內的所有頻率。
        :param mode: 'all' 表示返回所有頻率的樣本點。
        :param variables: 返回的樣本點包含的變量名稱。
        :return: LabelTensor 包含 x, y, z, f。
        """
        samples = []
        x, y, z = self.point

        for f in self.frequencies:
            samples.append([x, y, z, f])

        return LabelTensor(torch.tensor(samples, dtype=torch.float32, device=self.device), labels=variables)

    def __repr__(self):
        return (f"PortLocation(point={self.point}, frequencies={self.frequencies}, device='{self.device}')")


# 測試 3D PolygonLocation

if __name__ == "__main__":
    # 定義 XY 平面頂點
    scale = 1e-6
    vertices = [
        (1.0 * scale, -1.0 * scale),
        (-1.0 * scale, -1.0 * scale),
        (-1.0 * scale, 1.0 * scale),
        (1.0 * scale, 1.0 * scale)
    ]
    bound = {
        'x': [-1 * scale, 2 * scale],
        'y': [-1 * scale, 1 * scale],
        'z': [0 * scale, 0.1 * scale]
    }
    # 創建 PolygonLocation 物件
    polygon = PolygonLocation(vertices, bound, f_values=[1e9], sample_mode='outer', z_range=(0.0, 1.0))

    # 執行邊緣取樣
    edge_points = polygon.sample(1500)
    print(edge_points.device)
    edge_points = torch.tensor(edge_points)

    print("取樣點總數:", edge_points.shape)  # 應有大量點
    print("範例點 (前5個):", edge_points[:5])  # 顯示前五個點
    print(edge_points.device)
    # 3D 視覺化
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    ax.scatter(edge_points[:, 0], edge_points[:, 1], c='b', marker='o',alpha=0.5)

    ax.set_title('3D Edge Sampling: XY Planes and Faces')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()
