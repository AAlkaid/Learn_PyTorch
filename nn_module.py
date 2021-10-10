import torch
from torch import nn

class Zhenyu(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

zhenyuu = Zhenyu()
x = torch.tensor(1.0)
output = zhenyuu(x)

print(output)