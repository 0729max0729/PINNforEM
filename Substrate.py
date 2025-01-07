from matplotlib import pyplot as plt
from pina import Condition
from Equations import DielectricPotentialEquation, ConductorPotentialEquation
from Locations import ConductorLocation, DielectricLocation
from Materials import Material


class Substrate:
    """
    封裝金屬和介質，根據不同材料類型自動生成相應的條件。
    """
    def __init__(self, conductors, bound, material, z_range,f_values, device='cpu'):
        """
        :param conductors: List of ConductorLocation 對象。
        :param dielectric: DielectricLocation 對象。
        :param device: 運算設備 (CPU 或 CUDA)。
        """
        self.conductors = [ConductorLocation(conductor,bound=bound,f_values=f_values, sample_mode='interior',device=device, z_range=z_range) for conductor in conductors]
        self.bound = bound
        self.device = device
        self.material=material
        self.dielectric=DielectricLocation(conductors=self.conductors,bound=bound,sample_mode="outer",
                                           z_range=self.conductors[0].z_range,device=device,f_values=self.conductors[0].f_values)

    def generate_conditions(self):
        """
        根據材料類型生成相應的條件。
        :return: Dictionary of Conditions。
        """
        conditions = {}

        ## **1️⃣ 金屬條件**
        for i, conductor in enumerate(self.conductors):
            condition_name = f'conductor_{i+1}'
            conditions[condition_name] = Condition(
                location=conductor,
                equation=ConductorPotentialEquation(mu=self.material.mu,sigma=self.material.sigma)
            )

        ## **2️⃣ 介質條件**
        conditions['dielectric'] = Condition(
            location=self.dielectric,
            equation=DielectricPotentialEquation(mu=self.material.mu,epsilon=self.material.epsilon,tand=self.material.tand)
        )

        return conditions



if __name__ == "__main__":
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
        (2.5, 1.0),
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
        z_range=(0, 0.5),
        device='cpu'
    )

    metal2 = ConductorLocation(
        vertices=vertices_metal2,
        f_values=f_values,
        sample_mode='interior',
        z_range=(0, 0.5),
        device='cpu'
    )

    material = Material(
        name='Air',
        epsilon=8.85e-12,
        mu=1.256e-6,
        tand=0,
    )

    # 封裝到 Substrate 中
    substrate = Substrate(
        conductors=[metal1, metal2],
        bound=bound_dielectric,
        material=material,
        device='cpu'
    )

    # 生成條件
    conditions = substrate.generate_conditions()

    # 輸出條件
    for name, condition in conditions.items():
        print(f"📝 Condition Name: {name}")
        print(f"   - Location: {condition.location}")
        print(f"   - Equation: {condition.equation}")
        print('--------------------------------------------------')



    # 金屬與介質採樣
    samples_metal1 = metal1.sample(n=1000)
    samples_metal2 = metal2.sample(n=1000)
    samples_dielectric = dielectric.sample(n=2000)

    # 轉換為 numpy 進行繪圖
    points_metal1 = samples_metal1[:, :3].cpu().numpy()
    points_metal2 = samples_metal2[:, :3].cpu().numpy()
    points_dielectric = samples_dielectric[:, :3].cpu().numpy()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 金屬塊 1
    ax.scatter(
        points_metal1[:, 0], points_metal1[:, 1], points_metal1[:, 2],
        c='red', marker='o', s=1, label='Metal Block 1'
    )

    # 金屬塊 2
    ax.scatter(
        points_metal2[:, 0], points_metal2[:, 1], points_metal2[:, 2],
        c='blue', marker='o', s=1, label='Metal Block 2'
    )

    # 介質區域
    ax.scatter(
        points_dielectric[:, 0], points_dielectric[:, 1], points_dielectric[:, 2],
        c='green', marker='o', s=1, label='Dielectric'
    )

    # 標籤和標題
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Substrate: Metal Blocks and Dielectric')
    ax.legend()

    plt.show()

