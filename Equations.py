import torch
from pina.equation import Equation
from pina.operators import grad


class Maxwell3DEquation(Equation):
    def __init__(self, sigma=0.0, epsilon=8.85e-12, mu=1.256e-6):
        """
        Maxwell3DEquation class. This class enforces the solution to satisfy
        the Maxwell dynamic electric field equations in 3D.

        :param torch.float32 sigma: conductivity coefficient (default 0.0)
        :param torch.float32 epsilon: permittivity coefficient (default 8.85e-12)
        :param torch.float32 mu: permeability coefficient (default 1.256e-6)
        """
        self.sigma = sigma
        self.epsilon = epsilon
        self.mu = mu

        def equation(input_, output_):
            """
            Define the residuals of Maxwell's equations in 3D.
            """
            E_x = output_.extract(['E_x'])
            E_y = output_.extract(['E_y'])
            E_z = output_.extract(['E_z'])
            H_x = output_.extract(['H_x'])
            H_y = output_.extract(['H_y'])
            H_z = output_.extract(['H_z'])

            # 方程 1: ∂E_y/∂z - ∂E_z/∂y = -μ ∂H_x/∂t
            residual_1 = grad(E_y, input_, d='z') - grad(E_z, input_, d='y') + self.mu * grad(H_x, input_, d='t')

            # 方程 2: ∂E_z/∂x - ∂E_x/∂z = -μ ∂H_y/∂t
            residual_2 = grad(E_z, input_, d='x') - grad(E_x, input_, d='z') + self.mu * grad(H_y, input_, d='t')

            # 方程 3: ∂E_x/∂y - ∂E_y/∂x = -μ ∂H_z/∂t
            residual_3 = grad(E_x, input_, d='y') - grad(E_y, input_, d='x') + self.mu * grad(H_z, input_, d='t')

            # 方程 4: ∂H_y/∂z - ∂H_z/∂y - σ E_x - ε ∂E_x/∂t = 0
            if self.sigma > 1:
                residual_4 = (grad(H_y, input_, d='z') - grad(H_z, input_, d='y') - self.sigma * E_x - self.epsilon * grad(E_x, input_, d='t')) / self.sigma
            else:
                residual_4 = grad(H_y, input_, d='z') - grad(H_z, input_, d='y') - self.sigma * E_x - self.epsilon * grad(E_x, input_, d='t')

            # 方程 5: ∂H_z/∂x - ∂H_x/∂z - σ E_y - ε ∂E_y/∂t = 0
            if self.sigma > 1:
                residual_5 = (grad(H_z, input_, d='x') - grad(H_x, input_, d='z') - self.sigma * E_y - self.epsilon * grad(E_y, input_, d='t')) / self.sigma
            else:
                residual_5 = grad(H_z, input_, d='x') - grad(H_x, input_, d='z') - self.sigma * E_y - self.epsilon * grad(E_y, input_, d='t')

            # 方程 6: ∂H_x/∂y - ∂H_y/∂x - σ E_z - ε ∂E_z/∂t = 0
            if self.sigma > 1:
                residual_6 = (grad(H_x, input_, d='y') - grad(H_y, input_, d='x') - self.sigma * E_z - self.epsilon * grad(E_z, input_, d='t')) / self.sigma
            else:
                residual_6 = grad(H_x, input_, d='y') - grad(H_y, input_, d='x') - self.sigma * E_z - self.epsilon * grad(E_z, input_, d='t')

            return residual_1 + residual_2 + residual_3 + residual_4 + residual_5 + residual_6

        super().__init__(equation)



import torch
from pina.equation import Equation
from pina.operators import grad


class InterfaceEMFieldEquation(Equation):
    def __init__(self, epsilon_1, epsilon_2, mu_1, mu_2, normal_vector, sigma_1=0.0, sigma_2=0.0):
        """
        InterfaceEMFieldEquation class. This class enforces the interface
        conditions for both electric and magnetic fields between two media.

        :param float epsilon_1: Permittivity of medium 1.
        :param float epsilon_2: Permittivity of medium 2.
        :param float mu_1: Permeability of medium 1.
        :param float mu_2: Permeability of medium 2.
        :param torch.Tensor normal_vector: Normal vector at the interface (shape: [3]).
        :param float sigma_1: Conductivity of medium 1.
        :param float sigma_2: Conductivity of medium 2.
        """
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.normal_vector = normal_vector / torch.norm(normal_vector)  # 確保法向量是單位向量
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

        def equation(input_, output_):
            """
            Define the interface electric and magnetic field conditions.
            """
            # 提取電場分量
            E_x = output_.extract(['E_x'])
            E_y = output_.extract(['E_y'])
            E_z = output_.extract(['E_z'])

            # 提取磁場分量
            H_x = output_.extract(['H_x'])
            H_y = output_.extract(['H_y'])
            H_z = output_.extract(['H_z'])

            # 合併成 3D 張量
            E = torch.stack([E_x, E_y, E_z], dim=-1)
            H = torch.stack([H_x, H_y, H_z], dim=-1)

            ## 1️⃣ **電場邊界條件**

            # 法向電場分量
            E_n1 = torch.sum(E * self.normal_vector, dim=-1)
            E_n2 = torch.sum(E * self.normal_vector, dim=-1)

            # 切向電場分量
            E_t1 = torch.cross(E, self.normal_vector.expand_as(E), dim=-1)
            E_t2 = torch.cross(E, self.normal_vector.expand_as(E), dim=-1)

            # 法向電場連續性條件
            if self.sigma_1 > 1:
                normal_condition_E = (self.epsilon_1 * E_n1 - self.epsilon_2 * E_n2) / self.sigma_1
            else:
                normal_condition_E = self.epsilon_1 * E_n1 - self.epsilon_2 * E_n2

            # 切向電場連續性條件
            if self.sigma_1 > 1:
                tangential_condition_E = torch.norm(E_t1 - E_t2, dim=-1) / self.sigma_1
            else:
                tangential_condition_E = torch.norm(E_t1 - E_t2, dim=-1)

            ## 2️⃣ **磁場邊界條件**

            # 法向磁場分量
            H_n1 = torch.sum(H * self.normal_vector, dim=-1)
            H_n2 = torch.sum(H * self.normal_vector, dim=-1)

            # 切向磁場分量
            H_t1 = torch.cross(H, self.normal_vector.expand_as(H), dim=-1)
            H_t2 = torch.cross(H, self.normal_vector.expand_as(H), dim=-1)

            # 法向磁場連續性條件 (B_n1 = B_n2)
            normal_condition_H = H_n1 - H_n2

            # 切向磁場連續性條件 (1/μ * H_t1 = 1/μ * H_t2)
            tangential_condition_H = torch.norm(H_t1 / self.mu_1 - H_t2 / self.mu_2, dim=-1)

            ## 3️⃣ **返回損失函數**
            return (
                normal_condition_E +
                tangential_condition_E +
                normal_condition_H +
                tangential_condition_H
            )

        super().__init__(equation)






import torch
from pina.equation import Equation
from pina.operators import grad


class FrequencyChargeDensityEquation(Equation):
    def __init__(self, frequencies=None, amplitudes=None, phases=None):
        """
        Frequency-based Charge Density Equation.

        :param float epsilon_0: Permittivity of free space (F/m).
        :param list frequencies: List of frequencies for each sinusoidal component.
        :param list amplitudes: List of amplitudes for each sinusoidal component.
        :param list phases: List of phase shifts for each sinusoidal component.
        """
        self.frequencies = frequencies if frequencies is not None else [1.0]
        self.amplitudes = amplitudes if amplitudes is not None else [1.0]
        self.phases = phases if phases is not None else [0.0]

        def equation(input_, output_):
            """
            Define the divergence equation with frequency-based charge density.
            """
            # 提取電場分量
            E_x = output_.extract(['E_x'])
            E_y = output_.extract(['E_y'])
            E_z = output_.extract(['E_z'])

            # 提取時間變量 (t)
            t = input_.extract(['t'])

            # 分別計算每個電場分量的散度
            div_E_x = grad(output_, input_,components=['E_x'], d=['x'])
            div_E_y = grad(output_, input_,components=['E_y'], d=['y'])
            div_E_z = grad(output_, input_,components=['E_z'], d=['z'])

            # 將散度相加得到總散度
            divergence_E = div_E_x + div_E_y + div_E_z

            # 計算電荷密度 ρ(t) 使用頻率疊加
            rho = torch.zeros_like(t, device=t.device)
            for A, f, phi in zip(self.amplitudes, self.frequencies, self.phases):
                rho += A * torch.sin(2 * torch.pi * f * t + phi)

            # 返回殘差（左邊和右邊的差值）
            return (divergence_E - rho)

        super().__init__(equation)


