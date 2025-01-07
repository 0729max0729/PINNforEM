import torch
from pina.equation import FixedValue
from pina.problem import TimeDependentProblem, SpatialProblem
from pina.geometry import CartesianDomain
from pina.condition import Condition
from sympy.physics.units import frequency

from Equations import InitialConditionEquation

from Substrate import Substrate


class Maxwell3D(TimeDependentProblem, SpatialProblem):
    """
    Maxwell 3D Electromagnetic Problem in the frequency domain using potentials (phi, A).
    """

    # **1️⃣ 定義輸出變量**
    output_variables = ['phi_r', 'phi_i']
    # phi_r: 電勢的實部, phi_i: 電勢的虛部
    # A_x_r, A_y_r, A_z_r: 磁向量勢的實部
    # A_x_i, A_y_i, A_z_i: 磁向量勢的虛部

    # **2️⃣ 預設範圍**
    spatial_domain = {}
    temporal_domain = {}

    # **3️⃣ 初始化條件**
    conditions = {}
    def __init__(self, substrates,
                 ports,
                 spatial_domain,
                 frequency_domain):
        """
        Initialize Maxwell 3D Problem with a MaterialHandler.

        :param material_handler: An instance of MaterialHandler managing multiple materials.
        :param spatial_domain: Custom spatial domain {'x': [x_min, x_max], 'y': [y_min, y_max], 'z': [z_min, z_max]}.
        :param frequency_domain: Custom frequency domain {'f': [f_min, f_max]}.
        """
        self.substrates = substrates
        self.wave_port=ports

        self.spatial_domain = CartesianDomain(spatial_domain)
        self.frequency_domain = CartesianDomain(frequency_domain)

        # **5️⃣ 動態條件**
        self._dynamic_conditions = self._build_conditions()

        # **6️⃣ 合併條件**
        self.conditions = {**self.__class__.conditions, **self._dynamic_conditions}

        self.__class__.spatial_domain = CartesianDomain(spatial_domain)
        self.__class__.temporal_domain = CartesianDomain(frequency_domain)
        super().__init__()

    def _build_conditions(self):
        """
        Build dynamic conditions for Maxwell3D using potentials (V, A) and material interfaces.

        :return: Dictionary containing problem conditions.
        """
        conditions = {}

        ## **7️⃣ 材料條件**
        for substrate in self.substrates:
            substrate_conditions = substrate.generate_conditions()
            conditions.update(substrate_conditions)

        ## **8️⃣ 初始條件**
        for wave_port in self.wave_port:
            conditions[f'initial_phi_port_{wave_port.name}'] = wave_port.create_condition()



        return conditions




if __name__ == "__main__":

    device = torch.device('cuda')
    port = PortLocation((0, 0, 0.05), [1e9], device=device)
    wave = InitialConditionEquation()
    # 定義自定義範圍
    custom_spatial_domain = {
        'x': [-2, 2],
        'y': [-2, 2],
        'z': [0, 0.2]
    }
    custom_frequency_domain = {
        'f': [1e7, 1e9]
    }

    vertices_air = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0)
    ]
    air_location = PolygonLocation(vertices_air, f_values=[1e9], sample_mode='interior', device=device, z_range=(0.0, 0.1))

    # 定義銅區域
    vertices_copper = [
        (1.0, -1.0),
        (2.0, -1.0),
        (2.0, 1.0),
        (1.0, 1.0)
    ]
    copper_location = PolygonLocation(vertices_copper, f_values=[1e9], sample_mode='edges', device=device,
                                      z_range=(0.0, 0.1))

    # 定義材料
    material_air = Material(
        name='Air',
        epsilon=8.85e-12,
        sigma=0.0,
        mu=1.256e-6,
        location=air_location
    )

    material_copper = Material(
        name='Copper',
        epsilon=1,
        sigma=5.8e7,
        mu=1.256e-6,
        location=copper_location
    )

    # 建立 MaterialHandler
    material_handler = MaterialHandler([material_air,material_copper])

    # 創建 Maxwell3D 問題
    problem = Maxwell3D(
        material_handler=material_handler,
        spatial_domain=custom_spatial_domain,
        frequency_domain=custom_frequency_domain,
        port=port,
        wave=wave
    )

    problem.print_information()
    problem.discretise_domain(n=1000, mode='random', variables=['x', 'y', 'z', 'f'], locations='all')