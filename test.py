import torch
from pina import LabelTensor
from pina.geometry import CartesianDomain
from pina.model import FeedForward

from NN import MultiscaleFourierNet
from train import problem

# 假设您的模型架构如下：
model = MultiscaleFourierNet(input_dimension=3)

model.to("cuda")
# 加载权重
checkpoint = torch.load('checkpoints/best_model.ckpt')
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

points=problem.input_pts

model(points)