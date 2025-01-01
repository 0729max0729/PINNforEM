import torch
from pina import Condition
from pina.geometry import Location



class Material:
    """
    Material class representing material properties for 3D electromagnetics simulation.
    """

    def __init__(self, name, location: Location, tand=0, sigma=0, epsilon=1, mu=1.256e-6):
        self.name = name
        self.epsilon = epsilon
        self.sigma = sigma
        self.mu = mu
        self.location = location
        self.tand = tand





import torch
from pina import Condition
from Equations import ConductorPotentialEquation,DielectricPotentialEquation
from Locations import PolygonLocation


class MaterialHandler:
    """
    Manage multiple materials and automatically generate interior and interface conditions in 3D space.
    """
    def __init__(self, materials):
        self.materials = materials

    def _get_overlap_location(self, material1, material2):
        """
        計算兩個材料之間的邊緣重疊區域，返回一個新的 Location。
        :param material1: 第一個材料
        :param material2: 第二個材料
        :return: PolygonLocation 表示重疊區域，如果沒有重疊則返回 None
        """
        # **1️⃣ 檢查 Z 軸範圍重疊**
        z1_min, z1_max = material1.location.z_range
        z2_min, z2_max = material2.location.z_range

        z_overlap_min = max(z1_min, z2_min)
        z_overlap_max = min(z1_max, z2_max)

        if z_overlap_min > z_overlap_max:
            return None  # 沒有 Z 軸重疊範圍

        # **2️⃣ 檢查 XY 平面上的邊緣是否有重疊**
        edges1 = [
            (material1.location.vertices[i], material1.location.vertices[(i + 1) % len(material1.location.vertices)])
            for i in range(len(material1.location.vertices))
        ]
        edges2 = [
            (material2.location.vertices[i], material2.location.vertices[(i + 1) % len(material2.location.vertices)])
            for i in range(len(material2.location.vertices))
        ]

        overlap_edges = []
        for edge1 in edges1:
            for edge2 in edges2:
                overlap = self._find_edge_overlap(edge1, edge2)
                if overlap:
                    overlap_edges.append(overlap)

        if not overlap_edges:
            return None  # 沒有有效的邊緣重疊

        # **3️⃣ 計算重疊區域頂點 (在 z_overlap_min 和 z_overlap_max 平面上)**
        overlap_vertices_min = []
        overlap_vertices_max = []

        for segment in overlap_edges:
            for point in segment:
                overlap_vertices_min.append((point[0], point[1], z_overlap_min))
                overlap_vertices_max.append((point[0], point[1], z_overlap_max))

        # **4️⃣ 按照原始邊緣順序排列頂點**
        # 按照每個 segment 的順序依次添加到頂點列表中
        ordered_vertices = []
        for edge in overlap_edges:
            for point in edge:
                vertex_min = (point[0], point[1], z_overlap_min)
                vertex_max = (point[0], point[1], z_overlap_max)
                if vertex_min not in ordered_vertices:
                    ordered_vertices.append(vertex_min)
                if vertex_max not in ordered_vertices:
                    ordered_vertices.append(vertex_max)

        # **5️⃣ 去除重複頂點，但保留順序**
        seen = set()
        final_vertices = []
        for vertex in ordered_vertices:
            if vertex not in seen:
                final_vertices.append(vertex)
                seen.add(vertex)

        return PolygonLocation(
            vertices=[(x, y) for x, y, _ in final_vertices],  # 投影回 XY 平面
            sample_mode='edges',
            z_range=(z_overlap_min, z_overlap_max),
            f_values=material1.location.f_values,
            device=material2.location.device
        )

    def _find_edge_overlap(self, edge1, edge2):
        """
        找到兩條邊緣 (edge1, edge2) 的重疊部分（如果存在）。
        :param edge1: 第一條邊 (點1, 點2)
        :param edge2: 第二條邊 (點1, 點2)
        :return: 重疊線段的起點和終點 [(x_start, y_start), (x_end, y_end)] 或 None
        """
        (x1, y1), (x2, y2) = edge1
        (x3, y3), (x4, y4) = edge2

        # **1️⃣ 檢查是否平行**
        if (x2 - x1) * (y4 - y3) != (y2 - y1) * (x4 - x3):
            return None  # 不平行，無法重疊

        # **2️⃣ 檢查是否共線**
        cross_product_1 = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
        cross_product_2 = (x4 - x1) * (y2 - y1) - (y4 - y1) * (x2 - x1)
        if cross_product_1 != 0 or cross_product_2 != 0:
            return None  # 不共線，無法重疊

        # **3️⃣ 計算重疊範圍**
        if x1 == x2:  # 垂直線段，使用 y 軸範圍
            y_overlap_start = max(min(y1, y2), min(y3, y4))
            y_overlap_end = min(max(y1, y2), max(y3, y4))
            if y_overlap_start < y_overlap_end:  # 有有效重疊
                return [(x1, y_overlap_start), (x1, y_overlap_end)]
            else:
                return None  # 沒有有效重疊

        elif y1 == y2:  # 水平線段，使用 x 軸範圍
            x_overlap_start = max(min(x1, x2), min(x3, x4))
            x_overlap_end = min(max(x1, x2), max(x3, x4))
            if x_overlap_start < x_overlap_end:  # 有有效重疊
                return [(x_overlap_start, y1), (x_overlap_end, y1)]
            else:
                return None  # 沒有有效重疊

        else:  # 一般線段
            x_overlap_start = max(min(x1, x2), min(x3, x4))
            x_overlap_end = min(max(x1, x2), max(x3, x4))
            y_overlap_start = max(min(y1, y2), min(y3, y4))
            y_overlap_end = min(max(y1, y2), max(y3, y4))

            if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
                return [(x_overlap_start, y_overlap_start), (x_overlap_end, y_overlap_end)]
            else:
                return None  # 沒有有效重疊區域

        # **4️⃣ 檢查是否只是單點相交**
        if (x_overlap_start == x_overlap_end) or (y_overlap_start == y_overlap_end):
            return None  # 單點相交，無有效重疊

        return None

    def apply_equations(self):
        """
        Apply internal and interface conditions for each material.
        :return: Dictionary of conditions.
        """
        conditions = {}

        # **1️⃣ 處理內部條件**
        for material in self.materials:
            if material.location.sample_mode in ['interior', 'outer']:
                if material.sigma>1:
                    conditions[f'Conductor_{material.name}'] = Condition(
                        location=material.location,
                        equation=ConductorPotentialEquation(
                            sigma=material.sigma,
                            mu=material.mu
                        )
                    )
                else:
                    conditions[f'Dielectric_{material.name}'] = Condition(
                        location=material.location,
                        equation=DielectricPotentialEquation(
                            epsilon=material.epsilon,
                            mu=material.mu,
                            tand=material.tand
                        )
                    )
        '''
        # **2️⃣ 處理介面條件**
        for i, material in enumerate(self.materials):
            if material.location.sample_mode in ['edges', 'both']:
                for j, neighbor in enumerate(self.materials):
                    if i == j:
                        continue  # 跳過自身

                    overlap_location = self._get_overlap_location(material, neighbor)
                    if overlap_location:
                        normal_vectors = overlap_location.calculate_normal_vector()
                        for k, normal in enumerate(normal_vectors):
                            conditions[f'interface_{material.name}_to_{neighbor.name}_edge_{k}'] = Condition(
                                location=overlap_location,
                                equation=InterfacePotentialFieldEquation(
                                    epsilon_1=material.epsilon,
                                    epsilon_2=neighbor.epsilon,
                                    mu_1=material.mu,
                                    mu_2=neighbor.mu,
                                    normal_vector=normal,
                                    sigma_1=material.sigma,
                                    sigma_2=neighbor.sigma
                                )
                            )
        '''
        return conditions


if __name__ == "__main__":
    # 定義材料位置
    vertices_air = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0)
    ]
    air_location = PolygonLocation(vertices_air, sample_mode='interior', z_range=(0.0, 0.1),f_values=1)

    vertices_copper = [
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0)
    ]
    copper_location = PolygonLocation(vertices_copper, sample_mode='edges', z_range=(0.0, 0.1),f_values=1)

    # 定義材料
    material_air = Material('Air', epsilon=8.85e-12, sigma=0.0, mu=1.256e-6,tand=0, location=air_location)
    material_copper = Material('Copper', epsilon=1.0, sigma=5.8e7, mu=1.256e-6,tand=0, location=copper_location)

    # 建立 MaterialHandler
    material_handler = MaterialHandler([material_air, material_copper])
    conditions = material_handler.apply_equations()

    # 查看所有條件
    for name, condition in conditions.items():
        print(name, condition.location.vertices)

