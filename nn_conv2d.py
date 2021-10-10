import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
import ssl

from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context
dataset = torchvision.datasets.CIFAR10("../dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

# dataset = torchvision.datasets.DatasetFolder("dataset/cifar-10-batches-py")

dataloader = DataLoader(dataset, batch_size=64)


class zhenyu(nn.Module):
    def __init__(self):
        super(zhenyu, self).__init__()
        # // 彩色3层
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


zy = zhenyu()

step = 0
writer = SummaryWriter("../logs")
for data in dataloader:
    imgs, targets = data
    output = zy(imgs)
    print(imgs.shape)
    # 64 3 32 32
    writer.add_images("input", imgs, step)
    # 64 6 30 30

    output = torch.reshape(output, (-1, 3, 30, 30))
    print(output.shape)
    writer.add_images("output", output, step)

    step += 1
