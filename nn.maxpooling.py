import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader


import ssl


ssl._create_default_https_context = ssl._create_unverified_context

dataset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0 ,1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5 ,5))
#
# print(input.shape)

class zhenyu(nn.Module):
    def __init__(self):
        super(zhenyu, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)


    def forward(self, x):
        output = self.maxpool1(input)
        return output
zy = zhenyu()

writer = SummaryWriter(log_dir="../logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    writer.add_images("input", imgs, step)
    output = zy(imgs)
    writer.add_images("output", output, step)
    step += 1
writer.close()