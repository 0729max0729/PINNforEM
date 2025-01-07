import torch
from pina.geometry import Location
from pina.model import FeedForward, DeepONet
from pina.utils import LabelTensor
from torch.cuda import device

from Locations import ConductorLocation, DielectricLocation

# make model
branch_net = FeedForward(input_dimensions=3, output_dimensions=10)
trunk_net = FeedForward(input_dimensions=1, output_dimensions=10)
model = DeepONet(branch_net=branch_net,
                 trunk_net=trunk_net,
                 input_indeces_branch_net=['x','y','z'],
                 input_indeces_trunk_net=['f'],
                 reduction='+',
                 aggregator='*')


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from pina import LabelTensor

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
        z_range=(0, 1),
        device='cpu'
    )

    metal2 = ConductorLocation(
        vertices=vertices_metal2,
        f_values=f_values,
        sample_mode='interior',
        z_range=(0, 1),
        device='cpu'
    )

    # 定義介質區域
    dielectric = DielectricLocation(
        conductors=[metal1, metal2],
        bound=bound_dielectric,
        f_values=f_values,
        sample_mode='outer',
        z_range=(0, 1),
        device='cpu'
    )

    # 每個金屬塊採樣 1000 個點
    samples_metal1 = metal1.sample(n=1000)
    samples_metal2 = metal2.sample(n=1000)

    # 介質區域採樣 2000 個點
    samples_dielectric = dielectric.sample(n=2000)

    result=model.forward(samples_dielectric)
    print(result.shape)