import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_loader = DataLoader(test_data, batch_size=64,
                         shuffle=False,
                         num_workers=0, drop_last=True)

# # 第一张图片
# img, target = test_data[0]

# print(img.shape)
# print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1
writer.close()