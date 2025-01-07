import torch
from pina import Condition
from pina.geometry import Location



class Material:
    """
    Material class representing material properties for 3D electromagnetics simulation.
    """

    def __init__(self, name, tand=0, sigma=0, epsilon=1, mu=1.256e-6):
        self.name = name
        self.epsilon = epsilon
        self.sigma = sigma
        self.mu = mu
        self.tand = tand






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

