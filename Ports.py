import torch
from pina import Condition
from pina.geometry import Location
from Equations import InitialConditionEquation
from Locations import PortLocation


class WavePort:
    """
    WavePort 類，用於定義電勢初始條件的端口 (Port Location)。
    """
    def __init__(self, name, position, frequencies, phi_r_init=0.0, phi_i_init=0.0, device=torch.device('cpu')):
        """
        初始化 WavePort 類。

        :param str name: 端口名稱 (port1, port2, ...)
        :param tuple position: 端口的 (x, y, z) 坐標。
        :param list frequencies: 頻率列表。
        :param float phi_r_init: 電勢實部的初始值。
        :param float phi_i_init: 電勢虛部的初始值。
        :param str device: 計算設備 ('cpu' 或 'cuda')。
        """
        self.name = name
        self.position = position
        self.frequencies = frequencies
        self.phi_r_init = phi_r_init
        self.phi_i_init = phi_i_init
        self.device = device

        # 定義 PortLocation
        self.location = PortLocation(position, frequencies, device=device)

    def create_condition(self):
        """
        創建對應的條件。
        :return: PINA Condition 對象。
        """
        return Condition(
            location=self.location,
            equation=InitialConditionEquation(
                phi_r_init=self.phi_r_init,
                phi_i_init=self.phi_i_init
            )
        )
