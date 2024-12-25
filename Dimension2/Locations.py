import torch
from matplotlib import pyplot as plt
from pina.geometry import Location
from pina.utils import LabelTensor




class PolygonLocation(Location):
    """Implementation of a Polygon Domain with flexible sampling based on a sample_mode label."""

    def __init__(self, vertices, sample_mode='interior', time_range=(0, 1)):
        """
        :param vertices: List of (x, y) tuples representing polygon vertices.
        :param sample_mode: Sampling mode ('interior', 'edges', 'both').
        :param time_range: Tuple (t_min, t_max) representing the time range for sampling.
        """
        super().__init__()
        self.vertices = self._ensure_counter_clockwise(vertices)
        if sample_mode not in ['interior', 'edges', 'both']:
            raise ValueError("sample_mode must be 'interior', 'edges', or 'both'")
        self.sample_mode = sample_mode
        self.time_range = time_range  # e.g., (0, 1)

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
            x, y = point
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
        Sample points on the polygon edges.
        """
        edge_points = []
        edges = [(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]) for i in range(len(self.vertices))]

        total_length = sum(torch.sqrt((torch.tensor(p2[0]) - torch.tensor(p1[0])) ** 2 +
                                      (torch.tensor(p2[1]) - torch.tensor(p1[1])) ** 2)
                           for p1, p2 in edges)

        points_per_edge = [int(n * torch.sqrt((torch.tensor(p2[0]) - torch.tensor(p1[0])) ** 2 +
                                              (torch.tensor(p2[1]) - torch.tensor(p1[1])) ** 2) / total_length)
                           for p1, p2 in edges]

        for (p1, p2), num_points in zip(edges, points_per_edge):
            for i in range(num_points):
                t = i / max(num_points - 1, 1)
                x = p1[0] * (1 - t) + p2[0] * t
                y = p1[1] * (1 - t) + p2[1] * t
                edge_points.append([x, y])

        return edge_points

    def _sample_interior(self, n):
        """
        Sample points inside the polygon.
        """
        interior_points = []
        bbox_x = [min(p[0] for p in self.vertices), max(p[0] for p in self.vertices)]
        bbox_y = [min(p[1] for p in self.vertices), max(p[1] for p in self.vertices)]

        while len(interior_points) < n:
            x = torch.rand(1) * (bbox_x[1] - bbox_x[0]) + bbox_x[0]
            y = torch.rand(1) * (bbox_y[1] - bbox_y[0]) + bbox_y[0]
            point = torch.tensor([x.item(), y.item()])
            if self.is_inside([point])[0]:
                interior_points.append([x.item(), y.item()])

        return interior_points

    def calculate_normal_vector(self):
        """
        Automatically calculate the normal vector for edge segments.
        """
        normal_vectors = []
        edges = [(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]) for i in range(len(self.vertices))]

        for (p1, p2) in edges:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            normal = torch.tensor([-dy, dx], dtype=torch.float32)
            normal = normal / torch.norm(normal)
            normal_vectors.append(normal)

        return normal_vectors

    def sample(self, n, mode='random', variables=['x', 'y', 't']):
        """
        Sample points based on sample_mode, including time dimension.
        """
        sampled_points = []

        if self.sample_mode == 'interior':
            sampled_points = self._sample_interior(n)
        elif self.sample_mode == 'edges':
            sampled_points = self._sample_on_edges(n)
        elif self.sample_mode == 'both':
            edge_points = self._sample_on_edges(n // 2)
            interior_points = self._sample_interior(n // 2)
            sampled_points = edge_points + interior_points

        # 加入時間維度
        t_values = torch.rand(len(sampled_points)) * (self.time_range[1] - self.time_range[0]) + self.time_range[0]
        sampled_points_with_time = [(*point, t.item()) for point, t in zip(sampled_points, t_values)]

        return LabelTensor(torch.tensor(sampled_points_with_time), labels=['x', 'y', 't'])




if __name__ == "__main__":
    vertices = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),  # 凹陷點
        (0.0, 1.0)
    ]

    # Create the polygon location
    polygon = PolygonLocation(vertices,sample_mode='both')

    pts_poly = polygon.sample(1500)
    plt.scatter(pts_poly[:,0],pts_poly[:,1], label='Sampled Points', alpha=0.5)
    plt.show()