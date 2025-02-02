import torch
from pina.equation import Equation
from pina.operators import laplacian

import torch
from pina.equation import Equation

import torch
from pina.equation import Equation

from Shape_function import ShapeFunction


class ConductorPotentialEquation(Equation):
    """
    導體內電勢方程 (實部與虛部表示，考慮 ε 的虛部，假設電荷密度為 0)。
    引入形狀函數 S(x, y, z) 作為修正項。
    """

    def __init__(self, sigma=0.0, epsilon=8.85e-12, mu=1.256e-6, tand=0.0, shape_function=ShapeFunction()):
        """
        :param float sigma: 導電率
        :param float epsilon: 介電常數
        :param float mu: 磁導率
        :param float tand: 損耗正切
        :param shape_function: 形狀函數 S(x, y, z)
        """
        self.sigma = sigma
        self.epsilon_real = epsilon
        self.epsilon_imag = epsilon * tand
        self.mu = mu
        self.shape_function = shape_function  # 傳入形狀函數

        def equation(input_, output_):
            """
            定義導體內電位方程的殘差，假設電荷密度為 0。
            """
            # 提取變量
            phi_r = output_.extract(['phi_r'])  # 電勢實部
            phi_i = output_.extract(['phi_i'])  # 電勢虛部

            f = input_.extract(['f'])
            omega = 2 * torch.pi * f


            laplacian_r = laplacian(output_, input_, components=['phi_r'], d=['x', 'y', 'z'])



            laplacian_i = laplacian(output_, input_, components=['phi_i'], d=['x', 'y', 'z'])

            # 方程殘差
            residual_r = laplacian_r + omega * self.mu * self.sigma * phi_i
            residual_i = laplacian_i - omega * self.mu * self.sigma * phi_r

            residual = torch.sqrt(residual_r ** 2 + residual_i ** 2) / omega

            return residual

        super().__init__(equation)


import torch
from pina.equation import Equation

import torch
from pina.equation import Equation


class DielectricPotentialEquation(Equation):
    """
    介質內電勢方程 (實部與虛部表示，考慮 ε 的虛部，假設電荷密度為 0)。
    支援多個形狀函數 S(x-x₀, y-y₀, z-z₀) 的加權和。
    """

    def __init__(self, epsilon=8.85e-12, mu=1.256e-6, tand=0.0, shape_function=ShapeFunction(), centers=None):
        """
        :param float epsilon: 介電常數
        :param float mu: 磁導率
        :param float tand: 損耗正切
        :param shape_functions: 形狀函數列表 [S1, S2, ...]
        :param centers: 形狀函數的中心點列表 [(x₀, y₀, z₀), ...]
        """
        self.epsilon_real = epsilon
        self.epsilon_imag = epsilon * tand
        self.mu = mu
        self.shape_function = shape_function if shape_function is not None else []



        def equation(input_, output_):
            """
            定義介質內電位方程的殘差。
            """
            phi_r = output_.extract(['phi_r'])
            phi_i = output_.extract(['phi_i'])
            f = input_.extract(['f'])
            omega = 2 * torch.pi * f



            laplacian_r=laplacian(output_,input_,components=['phi_r'],d=['x','y','z'])
            # -------------------------------
            # 計算虛部的梯度和 Laplacian
            # -------------------------------


            laplacian_i=laplacian(output_,input_,components=['phi_i'],d=['x','y','z'])
            # -------------------------------
            # 計算殘差
            # -------------------------------
            residual_r = (
                    laplacian_r
                    + (self.mu * self.epsilon_real * omega ** 2 * phi_r)
                    - (self.mu * self.epsilon_imag * omega ** 2 * phi_i)
            )
            residual_i = (
                    laplacian_i
                    + (self.mu * self.epsilon_real * omega ** 2 * phi_i)
                    + (self.mu * self.epsilon_imag * omega ** 2 * phi_r)
            )

            # 合併殘差
            residual = torch.sqrt(residual_r ** 2 + residual_i ** 2)

            return residual

        super().__init__(equation)


class InitialConditionEquation(Equation):
    """
    定義初始條件方程，用於指定 phi (電勢) 和 A (磁向量勢) 的初始值。
    適用於頻域或時域 Maxwell 方程的初始條件設定。
    """

    def __init__(self, phi_r_init=1.0, phi_i_init=0.0):
        """
        :param phi_r_init: 電勢的實部初始值
        :param phi_i_init: 電勢的虛部初始值
        :param A_r_init: 磁向量勢的實部初始值 (A_x_r, A_y_r, A_z_r)
        :param A_i_init: 磁向量勢的虛部初始值 (A_x_i, A_y_i, A_z_i)
        """
        self.phi_r_init = phi_r_init
        self.phi_i_init = phi_i_init

        def equation(input_, output_):
            """
            定義初始條件殘差。
            """
            # 提取輸出變量
            phi_r = output_.extract(['phi_r'])
            phi_i = output_.extract(['phi_i'])


            # 計算殘差
            residual_phi_r = phi_r - self.phi_r_init
            residual_phi_i = phi_i - self.phi_i_init

            # 返回殘差總和
            return torch.sqrt(
                residual_phi_r**2 +
                residual_phi_i**2
            )*10

        super().__init__(equation)