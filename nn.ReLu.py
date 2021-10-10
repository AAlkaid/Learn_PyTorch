import torch

from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))

print(input.shape)

class zhenyu(nn.Module):
    def __init__(self):
        super(zhenyu, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output

zy = zhenyu()
output = zy(input)
print(output)

a = SummaryWriter("ss")