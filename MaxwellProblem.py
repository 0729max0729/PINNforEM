from matplotlib import pyplot as plt


from pina.problem import TimeDependentProblem, SpatialProblem
from pina.geometry import CartesianDomain
from pina.condition import Condition
from pina.equation import Equation, FixedValue
from sympy.physics.units import frequency

from Locations import PolygonLocation
from Materials import Material, MaterialHandler



from pina.problem import TimeDependentProblem, SpatialProblem
from pina.geometry import CartesianDomain
from pina import Condition
from Equations import Maxwell3DEquation, InterfaceEMFieldEquation, FrequencyChargeDensityEquation


class Maxwell3D(TimeDependentProblem, SpatialProblem):
    """
    Maxwell 3D Dynamic Electric Field Problem with MaterialHandler integration.
    """
    # 定義輸出變量
    output_variables = ['E_x', 'E_y', 'E_z', 'H_x', 'H_y', 'H_z']

    # 定義空間和時間域
    spatial_domain = CartesianDomain({'x': [-1, 1], 'y': [-1, 1], 'z': [0, 0.1]})
    temporal_domain = CartesianDomain({'t': [0, 1e-3]})

    # 初始化條件為空字典
    conditions = {}

    def __init__(self, material_handler):
        """
        Initialize Maxwell 3D Problem with a MaterialHandler.

        :param material_handler: An instance of MaterialHandler managing multiple materials.
        """
        self.material_handler = material_handler

        # 建立動態條件
        self._dynamic_conditions = self._build_conditions()

        # 將動態條件合併到靜態條件中
        self.conditions = {**self.__class__.conditions, **self._dynamic_conditions}

        super().__init__()

    def _build_conditions(self):
        """
        Build dynamic conditions for Maxwell3D based on materials and their interfaces.

        :return: Dictionary containing problem conditions.
        """
        conditions = {}

        # 應用 MaterialHandler 提供的條件
        material_conditions = self.material_handler.apply_equations()
        conditions.update(material_conditions)



        conditions['initial'] = Condition(
            location=CartesianDomain({'x': 0, 'y': 0, 'z': 0.05, 't': [0, 1e-3]}),
            equation=FrequencyChargeDensityEquation(frequencies=[1e5], amplitudes=[1], phases=[0])
        )


        return conditions




if __name__ == "__main__":


    vertices_air = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0)
    ]
    air_location = PolygonLocation(vertices_air, sample_mode='both')

    vertices_copper = [
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0)
    ]
    copper_location = PolygonLocation(vertices_copper, sample_mode='both')


    material_air = Material(
        name='Air',
        epsilon=8.85e-12,
        sigma=0.0,
        mu=1.256e-6,
        location=air_location
    )

    material_copper = Material(
        name='Copper',
        epsilon=1.0e-9,
        sigma=5.8e7,
        mu=1.256e-6,
        location=copper_location
    )

    # 創建 MaterialHandler
    material_handler = MaterialHandler([material_air, material_copper])



    problem = Maxwell3D(material_handler=material_handler)

    # 確認條件
    for name, condition in problem.conditions.items():
        print(f"{name}: {condition}")




    problem.discretise_domain(n=20, mode='grid', variables=['x','y','t'], locations='all')

    # 檢查取樣點是否正確分配到每個條件
    for key, points in problem.input_pts.items():
        print(f"{key}: {points.shape if points is not None else 'None'}")



    import matplotlib.pyplot as plt


    def plot_samples_normals_equations(material, color, label, equation_label):
        """
        繪製取樣點、邊界法向量，並標示方程名稱。
        """
        points = material.location.sample(n=100, mode='random', variables=['x', 'y'])
        vertices = material.location.vertices
        normals = material.location.calculate_normal_vector()

        # 繪製取樣點
        plt.scatter(points[:, 0], points[:, 1], color=color, alpha=0.5, label=f'{label} Samples')

        # 繪製多邊形邊界
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1)

            # 中點計算
            mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            normal = normals[i]

            # 繪製法向量箭頭
            plt.arrow(mid_point[0], mid_point[1],
                      normal[0] * 0.1, normal[1] * 0.1,
                      head_width=0.05, head_length=0.05, fc=color, ec=color)

        # 在區域內部標示方程
        center_x = sum(p[0] for p in vertices) / len(vertices)
        center_y = sum(p[1] for p in vertices) / len(vertices)
        plt.text(center_x, center_y, equation_label,
                 fontsize=10, color=color, ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.5'))


    # 創建視覺化圖表
    plt.figure(figsize=(12, 8))

    # 空氣區域
    plot_samples_normals_equations(
        material_air,
        color='skyblue',
        label='Air',
        equation_label='Maxwell2D\nσ=0, ε=8.85e-12, μ=1.256e-6'
    )

    # 銅區域
    plot_samples_normals_equations(
        material_copper,
        color='orange',
        label='Copper',
        equation_label='Maxwell2D\nσ=5.8e7, ε=1.0e-9, μ=1.256e-6'
    )

    # 介面
    interface_vertices = [
        (1.0, 0.0),
        (1.0, 1.0)
    ]
    for i in range(len(interface_vertices) - 1):
        p1 = interface_vertices[i]
        p2 = interface_vertices[i + 1]
        mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=2, label='Interface')
        plt.arrow(mid_point[0], mid_point[1],
                  0, 0.1,
                  head_width=0.05, head_length=0.05, fc='red', ec='red')
        plt.text(mid_point[0], mid_point[1] + 0.1, 'Interface\nε1 ≠ ε2',
                 fontsize=10, color='red', ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

    # 設定圖表細節
    plt.title('Material Samples, Normals, and Equations')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()



