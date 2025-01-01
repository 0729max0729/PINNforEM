import torch
from pina.model import FeedForward, FNO, AveragingNeuralOperator
from pina.solvers import PINN, SupervisedSolver

from pina import Trainer, Plotter

from pytorch_lightning.callbacks import ModelCheckpoint

from Equations import InitialConditionEquation
from Locations import PolygonLocation, PortLocation
from Materials import Material, MaterialHandler
from MaxwellProblem import Maxwell3D
from NN import TimeSpaceNet
from Ports import WavePort

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
f=[1]
scale=1
# 定義自定義範圍
custom_spatial_domain = {
    'x': [-2*scale, 2*scale],
    'y': [-2*scale, 2*scale],
    'z': [-1*scale, 1*scale],
}
custom_frequency_domain = {
    'f': [f[0],f[-1]]
}

bound=[custom_spatial_domain['x']]

# 定義兩個端口
port1 = WavePort(
    name='port1',
    position=(-2*scale, 0, 0),
    frequencies=f,
    phi_r_init=1.0,
    phi_i_init=0.0,
    device=device
)
# 定義兩個端口
port2 = WavePort(
    name='port2',
    position=(2*scale, 0, 0),
    frequencies=f,
    phi_r_init=-1.0,
    phi_i_init=0.0,
    device=device
)



# 定義銅區域
vertices_copper = [
    (1.0*scale, -1.0*scale),
    (-1.0*scale, -1.0*scale),
    (-1.0*scale, 1.0*scale),
    (1.0*scale, 1.0*scale)
]
copper_location = PolygonLocation(vertices_copper,custom_spatial_domain,f_values=f, sample_mode='interior',device=device, z_range=(-1*scale, 1*scale))
air_location = PolygonLocation(vertices_copper,custom_spatial_domain,f_values=f, sample_mode='outer',device=device, z_range=(-1*scale, 1*scale))
# 定義材料
material_air = Material(
    name='Air',
    epsilon=8.85e-12,
    mu=1.256e-6,
    tand=0,
    location=air_location
)

material_copper = Material(
    name='Copper',
    sigma=5.8e7,
    mu=1.256e-6,
    location=copper_location
)

# 建立 MaterialHandler
#material_handler = MaterialHandler([material_air])
material_handler = MaterialHandler([material_air,material_copper])






# 創建 Maxwell2D 問題
problem = Maxwell3D(spatial_domain=custom_spatial_domain,frequency_domain=custom_frequency_domain,material_handler=material_handler,ports=[port1,port2])

# 離散化網格
problem.discretise_domain(n=1000, mode='random', variables=['x', 'y', 'z', 'f'], locations='all')
problem.print_information()



# 定義神經網路模型
#model = TimeSpaceNet().to(device)


# make model
class SIREN(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)
model = FeedForward(
    layers=[10,10,10,10,10,10,10,10],
    func=torch.nn.Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
).to(device)











# 初始化 PINN
pinn = PINN(
    problem=problem,  # 3D Maxwell 問題
    model=model,
    extra_features=[],
    optimizer_kwargs={'lr': 1e-2},
    scheduler=torch.optim.lr_scheduler.MultiStepLR,
    scheduler_kwargs={'milestones' : [200, 500, 900, 1200], 'gamma':0.9}
)

# 定义模型检查点回调
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',  # 保存模型的目录
    filename='best_model',   # 保存的模型文件名
    save_top_k=5,            # 仅保存最好的模型
    monitor='mean_loss',     # 监控的指标
    mode='min'               # 指标越小越好
)

# 创建 Trainer 实例，并传入回调函数
trainer = Trainer(
    solver=pinn,
    max_epochs=800,
    callbacks=[checkpoint_callback],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    enable_model_summary=False
)

if __name__ == "__main__":
    # 開始訓練
    trainer.train()

    torch.save(model.state_dict(), "model.pth")
    # 可視化損失曲線
    plotter = Plotter()

    for ff in f:
        plotter.plot(
            solver=pinn,
            components='phi_r',
            fixed_variables={'z':0.05, 'f': ff},
            res=100,
            title=f'phi_r at f={ff}'
        )
    for ff in f:
        plotter.plot(
            solver=pinn,
            components='phi_i',
            fixed_variables={'z': 0.05, 'f': ff},
            res=100,
            title=f'phi_i at f={ff}'
        )
