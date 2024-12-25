import torch
from pina import Condition
from pina.geometry import Location
from pina.utils import LabelTensor
from pina.equation import Equation

from Equations import Maxwell2DEquation, InterfaceElectricFieldEquation
from Locations import PolygonLocation


import torch
from pina import Condition
from pina.geometry import Location
from pina.utils import LabelTensor
from Equations import Maxwell2DEquation, InterfaceElectricFieldEquation


class Material:
    """
    Material class representing material properties for electromagnetics simulation.

    Attributes:
        name (str): Name of the material.
        epsilon (float): Permittivity of the material (F/m).
        sigma (float): Conductivity of the material (S/m).
        mu (float): Permeability of the material (H/m).
        location (Location): Geometrical region where this material is applied.
    """

    def __init__(self, name, epsilon, sigma, mu, location: Location):
        """
        Initialize a material with its properties and associated location.

        :param name: Material name.
        :param epsilon: Permittivity of the material.
        :param sigma: Conductivity of the material.
        :param mu: Permeability of the material.
        :param location: A Location object defining the geometry of the material.
        """
        self.name = name
        self.epsilon = epsilon
        self.sigma = sigma
        self.mu = mu
        self.location = location

    def apply_to_equation(self, interface_material=None):
        """
        Apply material properties based on the location sample_mode.
        :param interface_material: Material on the other side of the interface.
        :return: A list of equations if edges are involved, otherwise a single equation.
        """
        if self.location.sample_mode == 'interior':
            # 内部区域应用 Maxwell 方程
            return Maxwell2DEquation(sigma=self.sigma, epsilon=self.epsilon, mu=self.mu)

        elif self.location.sample_mode == 'edges' and interface_material:
            # 边界区域：为每条边界分配一个 InterfaceElectricFieldEquation
            normal_vectors = self.location.calculate_normal_vector()
            edge_equations = []

            for normal_vector in normal_vectors:
                eq = InterfaceElectricFieldEquation(
                    epsilon_1=self.epsilon,
                    epsilon_2=interface_material.epsilon,
                    normal_vector=normal_vector,
                    sigma_1=self.sigma,
                    sigma_2=interface_material.sigma
                )
                edge_equations.append(eq)

            return edge_equations

        elif self.location.sample_mode == 'both' and interface_material:
            # 同时处理内部和边界
            normal_vectors = self.location.calculate_normal_vector()
            edge_equations = [
                InterfaceElectricFieldEquation(
                    epsilon_1=self.epsilon,
                    epsilon_2=interface_material.epsilon,
                    normal_vector=normal_vector,
                    sigma_1=self.sigma,
                    sigma_2=interface_material.sigma
                )
                for normal_vector in normal_vectors
            ]
            return {
                'interior': Maxwell2DEquation(sigma=self.sigma, epsilon=self.epsilon, mu=self.mu),
                'edges': edge_equations
            }
        else:
            raise ValueError("Unsupported sample_mode. Choose from 'interior', 'edges', 'both'.")




import torch
from pina import Condition
from Equations import Maxwell2DEquation, InterfaceElectricFieldEquation
from typing import List


class MaterialHandler:
    """
    管理多个材料，根据相邻材料自动生成内部和接口条件。
    """
    def __init__(self, materials: List):
        self.materials = materials

    def apply_equations(self):
        """
        为每个材料应用内部条件和接口条件。
        :return: Dictionary of conditions.
        """
        conditions = {}

        for material in self.materials:
            # 1. 应用内部条件
            if material.location.sample_mode in ['interior', 'both']:
                conditions[f'interior_{material.name}'] = Condition(
                    location=material.location,
                    equation=Maxwell2DEquation(
                        sigma=material.sigma,
                        epsilon=material.epsilon,
                        mu=material.mu
                    )
                )

            # 2. 应用接口条件（检查相邻材料）
            if material.location.sample_mode in ['edges', 'both']:
                for neighbor in self.materials:
                    if material == neighbor:
                        continue

                    normal_vectors = material.location.calculate_normal_vector()
                    for i, normal in enumerate(normal_vectors):
                        conditions[f'interface_{material.name}_to_{neighbor.name}_edge_{i}'] = Condition(
                            location=material.location,
                            equation=InterfaceElectricFieldEquation(
                                epsilon_1=material.epsilon,
                                epsilon_2=neighbor.epsilon,
                                normal_vector=normal,
                                sigma_1=material.sigma,
                                sigma_2=neighbor.sigma
                            )
                        )

        return conditions






if __name__ == "__main__":
    # 定義空氣區域
    vertices_air = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0)
    ]
    air_location = PolygonLocation(vertices_air, sample_mode='both')

    # 定義介面區域 (Interface)
    vertices_interface = [
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 0.5),
        (1.0, 0.5)
    ]
    interface_location = PolygonLocation(vertices_interface, sample_mode='both')

    # 定義銅區域
    vertices_copper = [
        (1.0, 0.5),
        (2.0, 0.5),
        (2.0, 1.0),
        (1.0, 1.0)
    ]
    copper_location = PolygonLocation(vertices_copper, sample_mode='both')

    # 定義材料
    material_air = Material(
        name='Air',
        epsilon=8.85e-12,
        sigma=0.0,
        mu=1.256e-6,
        location=air_location
    )

    material_interface = Material(
        name='Interface',
        epsilon=5.0e-11,
        sigma=1.0e4,
        mu=1.256e-6,
        location=interface_location
    )

    material_copper = Material(
        name='Copper',
        epsilon=1.0e-9,
        sigma=5.8e7,
        mu=1.256e-6,
        location=copper_location
    )

    # 創建 MaterialHandler
    material_handler = MaterialHandler([material_air, material_interface, material_copper])

    # 應用方程
    conditions = material_handler.apply_equations()

    # 檢視所有條件
    for name, condition in conditions.items():
        print(f"{name}: {condition}")

    import matplotlib.pyplot as plt

    # 取樣點
    air_points = material_air.location.sample(200)
    interface_points = material_interface.location.sample(200)
    copper_points = material_copper.location.sample(200)

    plt.figure(figsize=(10, 8))
    plt.scatter(air_points[:, 0], air_points[:, 1], label='Air Region', alpha=0.5, color='blue')
    plt.scatter(interface_points[:, 0], interface_points[:, 1], label='Interface Region', alpha=0.5, color='green')
    plt.scatter(copper_points[:, 0], copper_points[:, 1], label='Copper Region', alpha=0.5, color='orange')

    plt.title('Material Regions and Interfaces')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


    def plot_normals(location, label, color):
        vertices = location.vertices
        normals = location.calculate_normal_vector()

        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]

            # 中點
            mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            # 法向量
            normal = normals[i]

            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1)
            plt.arrow(mid_point[0], mid_point[1],
                      normal[0] * 0.1, normal[1] * 0.1,
                      head_width=0.05, head_length=0.05, fc=color, ec=color)

        plt.scatter([v[0] for v in vertices], [v[1] for v in vertices], color=color, label=label)


    # 視覺化法向量
    plt.figure(figsize=(10, 8))
    plot_normals(air_location, label='Air Normals', color='blue')
    plot_normals(interface_location, label='Interface Normals', color='green')
    plot_normals(copper_location, label='Copper Normals', color='orange')
    plt.title('Normal Vectors on Material Interfaces')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


