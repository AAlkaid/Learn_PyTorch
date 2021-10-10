from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# use transform.ToTensor to handle two problems
# 1. How to use transform?
# 2. What the difference between tensor and the normal data types?
# 3. Why use tensor?

# absolute path
# /Users/buzhenyu/PycharmProjects/learn_PyTorch/dataset/train/ants/0013035.jpg
img_path = "dataset/train/ants/0013035.jpg"
# img_path_abs = "/Users/buzhenyu/PycharmProjects/learn_PyTorch/dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1.
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
