from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/kbd.jpg")

# to tensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("ToTensor", img_tensor)


# normalization
print(img_tensor[0][0][0])
# trans_norm = transforms.Normalize([0.5, 0.5, 0.5],
#                                   [0.5, 0.5, 0.5])

trans_norm = transforms.Normalize([1, 3, 5],
                                  [5, 5, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])

writer.add_image("Normalize", img_norm)


# resize

print(img.size)
trans_resize = transforms.Resize((300, 300))

# img PIL --> resize --> img_resize PIL
img_resize = trans_resize(img)
# img PIL --> totensor --> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)


# compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensortrans_totensor
trans_compose = transforms.Compose([trans_totensor, trans_resize_2])
img_resize_2 = trans_compose(img)
writer.add_image("Resize1", img_resize_2, 1)


# randomcrop
trans_random = transforms.RandomCrop(200, 100)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])

for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()
