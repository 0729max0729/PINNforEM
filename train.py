import torch
from pina.solvers import PINN

from pina import Trainer, Plotter

from pytorch_lightning.callbacks import ModelCheckpoint


from Locations import PolygonLocation
from Materials import Material, MaterialHandler
from MaxwellProblem import Maxwell3D
from NN import MultiscaleFourierNet, SinCosFeature3D

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定義空氣區域
vertices_air = [
    (0.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (0.0, 1.0)
]
air_location = PolygonLocation(vertices_air, sample_mode='both',device=device, z_range=(0.0, 0.1),time_range=(0, 1e-3))

# 定義銅區域
vertices_copper = [
    (1.0, -1.0),
    (2.0, -1.0),
    (2.0, 1.0),
    (1.0, 1.0)
]
copper_location = PolygonLocation(vertices_copper, sample_mode='both',device=device, z_range=(0.0, 0.1),time_range=(0, 1e-3))

# 定義材料
material_air = Material(
    name='Air',
    epsilon=8.85e-12,
    sigma=0.0,
    mu=1.256e-6,
    location=air_location
)

material_copper = Material(
    name='Copper',
    epsilon=1,
    sigma=5.8e7,
    mu=1.256e-6,
    location=copper_location
)

# 建立 MaterialHandler
material_handler = MaterialHandler([material_air])







# 創建 Maxwell2D 問題
problem = Maxwell3D(material_handler=material_handler)

# 離散化網格
problem.discretise_domain(n=10000, mode='random', variables=['x', 'y', 'z', 't'], locations='all')
# 檢查每個條件的標籤
for name, points in problem.input_pts.items():
    if points is not None:
        print(f"{name}: {points.labels}")
    else:
        print(f"{name}: ❌ No points sampled")



# 定義神經網路模型
model = MultiscaleFourierNet(input_dimension=4,output_dimension=6)


# 初始化 PINN
pinn = PINN(
    problem=problem,  # 3D Maxwell 問題
    model=model,
    optimizer_kwargs={'lr': 0.001}
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
    max_epochs=1000,
    callbacks=[checkpoint_callback],
    accelerator='gpu',
    enable_model_summary=True
)

if __name__ == "__main__":
    # 開始訓練
    trainer.train()


    # 可視化損失曲線
    plotter = Plotter()

    # 提取 E_x
    plotter.plot(
        solver=pinn,
        components='E_x',
        fixed_variables={'t': 1e-2,'z':0.05},
        method='pcolor',
        res=1000,
        title='Electric Field E_x at t=0'
    )

    # 提取 E_y
    plotter.plot(
        solver=pinn,
        components='E_y',
        fixed_variables={'t': 1e-2,'z':0.05},
        method='contourf',
        res=1000,
        title='Electric Field E_y at t=0'
    )

    # 提取 H_z
    plotter.plot(
        solver=pinn,
        components='H_z',
        fixed_variables={'t': 1e-2,'z':0.05},
        method='contourf',
        res=1000,
        title='Magnetic Field H_z at t=0'
    )

