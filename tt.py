# 導入必要的模組
import torch
from pina import Condition, Trainer, Plotter
from pina.model import FeedForward
from pina.problem import SpatialProblem
from pina.operators import grad
from pina.geometry import  CartesianDomain
from pina.equation import Equation, FixedValue
from pina.solvers import PINN


# 定義問題類別
class SimpleODE(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1]})

    # 定義微分方程
    def ode_equation(input_, output_):
        x=input_.extract(['x'])
        u_x = torch.sin(2*torch.pi*5*x)
        u = output_.extract(['u'])
        return u_x - u

    # 定義初始和邊界條件
    conditions = {
        'x0': Condition(
            location=CartesianDomain({'x': 0.}),
            equation=FixedValue(1)
        ),
        'D': Condition(
            location=CartesianDomain({'x': [0, 1]}),
            equation=Equation(ode_equation)
        ),
    }

    # 定義真實解，用於驗證
    def truth_solution(self, pts):
        x=pts.extract(['x'])
        return torch.sin(2*torch.pi*5*x)


problem=SimpleODE()
# 初始化模型
model = FeedForward(
    layers=[10, 10],
    func=torch.nn.Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
)

# 初始化 PINN
pinn = PINN(
    problem=problem,
    model=model,
    extra_features=[],
    optimizer_kwargs={'lr': 0.001, 'weight_decay': 1e-6}
)
problem.discretise_domain(1, 'random', locations=['x0'])
problem.discretise_domain(20, 'lh', locations=['D'])
# 初始化訓練器
trainer = Trainer(
    pinn,
    max_epochs=10000,
    accelerator='cpu',
    enable_model_summary=False
)

# 開始訓練
trainer.train()

# 驗證結果
plotter = Plotter()
plotter.plot(solver=pinn)
