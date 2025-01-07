import torch


def laplacian_autograd(phi, coords):
    """
    使用 PyTorch Autograd 计算函数 phi 的 Laplacian。
    :param phi: Tensor, 需要计算 Laplacian 的函数值。
    :param coords: Tensor, 坐标张量，包含 ['x', 'y', 'z']。
    :return: Tensor, Laplacian 结果。
    """
    grad = torch.autograd.grad(
        outputs=phi,
        inputs=coords,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True
    )[0]  # 返回梯度

    # 分别对每个分量再次求导，得到二阶偏导数
    laplacian = 0.0
    for i in range(coords.shape[1]):
        second_derivative = torch.autograd.grad(
            outputs=grad[:, i],
            inputs=coords,
            grad_outputs=torch.ones_like(grad[:, i]),
            create_graph=True,
            retain_graph=True
        )[0][:, i]
        laplacian += second_derivative

    return laplacian
