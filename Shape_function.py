import torch


class ShapeFunction(torch.nn.Module):
    def forward(self, coords):
        return torch.exp(-torch.sum(coords ** 2, dim=1))
