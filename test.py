import torch
from pina.solvers import PINN
from pina.utils import LabelTensor

from NN import SinCosFeature3D, MultiscaleFourierNet
from train import pinn, problem, model


def get_solution_data(solver, res=50, fixed_variables=None, components=None):
    """
    获取 PINA 模型的预测数据，而不是直接绘图。

    :param solver: PINA 求解器实例。
    :param res: 每个维度的采样分辨率。
    :param fixed_variables: 字典，指定某些变量的固定值（例如 {'z': 0.05, 't': 0.1}）。
    :param components: 要提取的输出变量（例如 'E_x', 'phi'）。
    :return: 一个包含输入点和预测结果的字典。
    """
    if fixed_variables is None:
        fixed_variables = {}
    if components is None:
        components = solver.problem.output_variables

    if isinstance(components, str):
        components = [components]

    if not isinstance(components, list):
        raise NotImplementedError("Output variables must be a string or list of strings.")

    # 选择要离散化的变量
    input_vars = [
        var for var in solver.problem.input_variables
        if var not in fixed_variables.keys()
    ]

    # 生成样本点 (网格采样)
    sampled_points = solver.problem.domain.sample(res, mode="grid", variables=input_vars)

    # 添加固定变量
    fixed_pts = torch.ones(sampled_points.shape[0], len(fixed_variables))
    fixed_pts *= torch.tensor(list(fixed_variables.values()))
    fixed_pts = fixed_pts.as_subclass(LabelTensor)
    fixed_pts.labels = list(fixed_variables.keys())

    sampled_points = sampled_points.append(fixed_pts)
    sampled_points = sampled_points.to(device=solver.device)

    # 进行预测
    predicted_output = solver.forward(sampled_points).extract(components)
    predicted_output = predicted_output.as_subclass(torch.Tensor).cpu().detach()
    sampled_points = sampled_points.cpu()

    # 将数据转换为字典形式返回
    data = {
        'input_points': sampled_points.numpy(),
        'predicted_output': predicted_output.numpy(),
        'input_labels': sampled_points.labels,
        'output_labels': components
    }

    return data


if __name__ == "__main__":
    # 固定时间和Z轴
    fixed_vars = {'x':0,'y':0,'z': 0.05}
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # 初始化 PINN
    pinn = PINN(
        problem=problem,  # 3D Maxwell 問題
        model=model,
        extra_features=[SinCosFeature3D()]
    )

    data = get_solution_data(pinn, res=1000, fixed_variables=fixed_vars, components='phi')

    # 查看提取的数据
    print("输入点 (Input Points):")
    print(data['input_points'])

    print("\n预测结果 (Predicted Output):")
    print(data['predicted_output'])

    print("\n输入标签 (Input Labels):", data['input_labels'])
    print("输出标签 (Output Labels):", data['output_labels'])

    import matplotlib.pyplot as plt

    # 假设是二维输入点
    t = data['input_points'][:, 0]

    phi = data['predicted_output']

    plt.plot(t, phi)
    plt.xlabel(data['input_labels'][0])
    plt.title('Predicted phi')
    plt.show()

