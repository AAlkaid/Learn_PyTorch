
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset/train/ants/24335309_c5ea483bb8.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')
# y = x
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()