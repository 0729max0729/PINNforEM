import torch
from pina import LabelTensor
from pina.equation import Equation
from pina.operators import grad


class Maxwell2DEquation(Equation):
    def __init__(self, sigma=0.0, epsilon=8.85e-12, mu=1.256e-6):
        """
        Maxwell2DEquation class. This class enforces the solution to satisfy
        the Maxwell dynamic electric field equations in 2D.

        :param torch.float32 sigma: conductivity coefficient (default 0.0)
        :param torch.float32 epsilon: permittivity coefficient (default 8.85e-12)
        :param torch.float32 mu: permeability coefficient (default 1.256e-6)
        """
        self.sigma = sigma
        self.epsilon = epsilon
        self.mu = mu

        def equation(input_, output_):
            """
            Define the residuals of Maxwell's equations.
            """
            E_x = output_.extract(['E_x'])
            E_y = output_.extract(['E_y'])
            H_z = output_.extract(['H_z'])

            # ∂E_y/∂x - ∂E_x/∂y = -μ ∂H_z/∂t (不涉及 σ，无需标准化)
            residual_1 = grad(E_y, input_, d='x') - grad(E_x, input_, d='y') + self.mu * grad(H_z, input_, d='t')

            # ∂H_z/∂y - σ E_x - ε ∂E_x/∂t = 0 (涉及 σ，需要标准化)
            if self.sigma > 1:
                residual_2 = (grad(H_z, input_, d='y') - self.sigma * E_x - self.epsilon * grad(E_x, input_, d='t')) / self.sigma
            else:
                residual_2 = grad(H_z, input_, d='y') - self.sigma * E_x - self.epsilon * grad(E_x, input_, d='t')

            # -∂H_z/∂x - σ E_y - ε ∂E_y/∂t = 0 (涉及 σ，需要标准化)
            if self.sigma > 1:
                residual_3 = (-grad(H_z, input_, d='x') - self.sigma * E_y - self.epsilon * grad(E_y, input_, d='t')) / self.sigma
            else:
                residual_3 = -grad(H_z, input_, d='x') - self.sigma * E_y - self.epsilon * grad(E_y, input_, d='t')

            return residual_1 ** 2 + residual_2 ** 2 + residual_3 ** 2

        super().__init__(equation)


class InterfaceElectricFieldEquation(Equation):
    def __init__(self, epsilon_1, epsilon_2, normal_vector, sigma_1=0.0, sigma_2=0.0):
        """
        InterfaceElectricFieldEquation class. This class enforces the interface
        conditions for the electric field between two media.

        :param float epsilon_1: Permittivity of medium 1
        :param float epsilon_2: Permittivity of medium 2
        :param torch.Tensor normal_vector: Normal vector at the interface (shape: [1, 2])
        :param float sigma_1: Conductivity of medium 1
        :param float sigma_2: Conductivity of medium 2
        """
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.normal_vector = normal_vector
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

        def equation(input_, output_):
            """
            Define the interface electric field conditions.
            """
            # 提取电场分量
            E_x = output_.extract(['E_x'])
            E_y = output_.extract(['E_y'])

            # 计算法向电场分量 (dot product with normal vector)
            E_n1 = E_x * self.normal_vector[0] + E_y * self.normal_vector[1]
            E_n2 = E_x * self.normal_vector[0] + E_y * self.normal_vector[1]

            # 计算切向电场分量 (cross product with normal vector)
            E_t1 = -E_x * self.normal_vector[1] + E_y * self.normal_vector[0]
            E_t2 = -E_x * self.normal_vector[1] + E_y * self.normal_vector[0]

            # 法向电场连续性条件（对涉及 σ 的部分进行标准化）
            if self.sigma_1 > 1:
                normal_condition = (self.epsilon_1 * E_n1 - self.epsilon_2 * E_n2) / self.sigma_1
            else:
                normal_condition = self.epsilon_1 * E_n1 - self.epsilon_2 * E_n2

            # 切向电场连续性条件（对涉及 σ 的部分进行标准化）
            if self.sigma_1 > 1:
                tangential_condition = (E_t1 - E_t2) / self.sigma_1
            else:
                tangential_condition = E_t1 - E_t2

            return normal_condition ** 2 + tangential_condition ** 2

        super().__init__(equation)




