import torch
from pina.model import DeepONet, FeedForward
from pina.solvers import PINN, SupervisedSolver, CompetitivePINN, RBAPINN

from pina import Trainer, Plotter

from pytorch_lightning.callbacks import ModelCheckpoint

from Locations import ConductorLocation, DielectricLocation
from Materials import Material
from MaxwellProblem import Maxwell3D
from NN import TimeSpaceNet, DEEPONET
from Ports import WavePort
from Substrate import Substrate
from extra_feature import HelmholtzFeature
from tt import model

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
f=[5e6]
scale=1
# 定義自定義範圍
custom_spatial_domain = {
    'x': [-5*scale, 5*scale],
    'y': [-2*scale, 2*scale],
    'z': [-0.1*scale, 0.1*scale],
}
custom_frequency_domain = {
    'f': [f[0],f[-1]]
}

bound=[custom_spatial_domain['x']]

# 定義兩個端口
port1 = WavePort(
    name='port1',
    position=(-0.1*scale, 0, 0),
    frequencies=f,
    phi_r_init=1,
    phi_i_init=0.0,
    device=device
)
# 定義兩個端口
port2 = WavePort(
    name='port2',
    position=(0.1*scale, 0, 0),
    frequencies=f,
    phi_r_init=-1,
    phi_i_init=0.0,
    device=device
)



# 定義銅區域
vertices_copper1 = [
    (-0.1*scale, -0.1*scale),
    (-1*scale, -0.1*scale),
    (-1*scale, 0.1*scale),
    (-0.1*scale, 0.1*scale)
]

vertices_copper2 = [
    (0.1*scale, -0.1*scale),
    (1*scale, -0.1*scale),
    (1*scale, 0.1*scale),
    (0.1*scale, 0.1*scale)
]
# 定義材料
material = Material(
    name='sub1',
    epsilon=8.85e-12,
    mu=1.256e-6,
    sigma=5.8e7,
    tand=0
)



# 建立 MaterialHandler
#material_handler = MaterialHandler([material_air])
substrate1=Substrate([vertices_copper1,vertices_copper2],custom_spatial_domain,material=material,z_range=(custom_spatial_domain['z'][0],custom_spatial_domain['z'][1]),f_values=f,device=device)






# 創建 Maxwell2D 問題
problem = Maxwell3D(spatial_domain=custom_spatial_domain,frequency_domain=custom_frequency_domain,substrates=[substrate1],ports=[port1,port2])

# 離散化網格
problem.discretise_domain(n=5000, mode='random', variables=['x', 'y', 'z', 'f'], locations='all')




# 定義神經網路模型
model = TimeSpaceNet().to(device)
#model = DEEPONET().to(device)
# make model

'''
# make model
class SIREN(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)
model = FeedForward(
    layers=[10,10,10,10,10,10,10,10],
    func=SIREN,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
).to(device)
'''










# 初始化 PINN
pinn = RBAPINN(
    problem=problem,  # 3D Maxwell 問題
    model=model,
    optimizer_kwargs={'lr': 1e-5},
    extra_features=[HelmholtzFeature(num_sources=2)]
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
    max_epochs=10,
    #callbacks=[checkpoint_callback],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    enable_model_summary=False
)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
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
            levels = 50,
            title=f'phi_r at f={ff}'
        )
    for ff in f:
        plotter.plot(
            solver=pinn,
            components='phi_i',
            fixed_variables={'z': 0.05, 'f': ff},
            levels=50,
            title=f'phi_i at f={ff}'
        )
