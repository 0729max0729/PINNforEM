import torch
from pina.geometry import Location
from pina.utils import LabelTensor
from torch.cuda import device


class PolygonLocation(Location):
    """
    Implementation of a 3D Polygon Domain with flexible sampling based on a sample_mode label,
    including normal vectors in both XY plane and Z direction.
    """

    def __init__(self, vertices, sample_mode='interior', z_range=(0, 1), time_range=(0, 1),device='cpu'):
        """
        :param vertices: List of (x, y) tuples representing polygon vertices (base).
        :param sample_mode: Sampling mode ('interior', 'edges', 'both').
        :param z_range: Tuple (z_min, z_max) representing the Z-axis range.
        :param time_range: Tuple (t_min, t_max) representing the time range for sampling.
        """
        super().__init__()
        self.vertices = self._ensure_counter_clockwise(vertices)
        if sample_mode not in ['interior', 'edges', 'both']:
            raise ValueError("sample_mode must be 'interior', 'edges', or 'both'")
        self.sample_mode = sample_mode
        self.z_range = z_range  # Z 軸範圍
        self.time_range = time_range  # 時間範圍

        self.device = device
    def _ensure_counter_clockwise(self, vertices):
        """
        Ensure the vertices are ordered counter-clockwise.
        """
        area = 0.0
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            area += (x2 - x1) * (y2 + y1)

        if area > 0:  # If area is positive, the vertices are clockwise
            vertices.reverse()
        return vertices

    def is_inside(self, points):
        """
        Check if points are inside the polygon using ray-casting algorithm.
        """
        result = []
        for point in points:
            x, y = point[:2]  # 只考慮 X, Y 平面
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

        return torch.tensor(result, dtype=torch.bool)



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
            while sampled_points_xy < num_points_per_section:
                x = torch.rand(1).item() * (bbox_x[1] - bbox_x[0]) + bbox_x[0]
                y = torch.rand(1).item() * (bbox_y[1] - bbox_y[0]) + bbox_y[0]
                point = torch.tensor([x, y])
                if self.is_inside([point])[0]:
                    edge_points.append([x, y, z])
                    sampled_points_xy += 1



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
            z = torch.rand(1).item() * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
            point = torch.tensor([x.item(), y.item()])
            if self.is_inside([point])[0]:
                interior_points.append([x.item(), y.item(), z])

        return interior_points

    def calculate_normal_vector(self):
        """
        Automatically calculate the normal vector for edge segments in 3D space.
        """
        normal_vectors = []
        edges = [(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]) for i in range(len(self.vertices))]
        z_min, z_max = self.z_range

        # 計算 XY 平面上的邊法向量
        for (p1, p2) in edges:
            # 定義兩個邊緣點 (p1, p2) 和 z 軸上下界，構建三個點來定義一個平面
            point1 = torch.tensor([p1[0], p1[1], z_min], dtype=torch.float32,device=self.device)
            point2 = torch.tensor([p2[0], p2[1], z_min], dtype=torch.float32,device=self.device)
            point3 = torch.tensor([p1[0], p1[1], z_max], dtype=torch.float32,device=self.device)

            # 計算兩個邊緣向量
            edge_vector_1 = point2 - point1  # 邊的方向向量
            edge_vector_2 = point3 - point1  # 垂直於邊的 z 軸方向向量

            # 計算叉積以獲得法向量
            normal = torch.cross(edge_vector_1, edge_vector_2, dim=0)
            normal = normal / torch.norm(normal)  # 正規化為單位向量

            normal_vectors.append(normal)

        # 添加平行於 Z 軸的法向量（上表面和下表面）
        top_surface_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32,device=self.device)  # Z 軸正方向
        bottom_surface_normal = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32,device=self.device)  # Z 軸負方向

        # 將上表面和下表面的法向量分別添加到法向量列表中

        normal_vectors.append(top_surface_normal)
        normal_vectors.append(bottom_surface_normal)

        return normal_vectors

    def sample(self, n, mode='random', variables=['x', 'y', 'z', 't']):
        """
        Sample points in 3D space, including Z and time dimensions.
        """


        if self.sample_mode == 'interior':
            sampled_points = self._sample_interior(n)
        elif self.sample_mode == 'edges':
            sampled_points = self._sample_on_edges(n)
        elif self.sample_mode == 'both':
            edge_points = self._sample_on_edges(n // 2)
            interior_points = self._sample_interior(n // 2)
            sampled_points = edge_points + interior_points



        # 添加 t 軸取樣
        t_values = torch.rand(len(sampled_points)) * (self.time_range[1] - self.time_range[0]) + self.time_range[0]

        # 合併 x, y, z, t
        sampled_points_with_t = [
            (point[0], point[1], point[2], t.item()) for point, t in zip(sampled_points, t_values)
        ]

        return LabelTensor(torch.tensor(sampled_points_with_t, dtype=torch.float32), labels=['x', 'y', 'z', 't'])