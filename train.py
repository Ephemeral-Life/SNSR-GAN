import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from SNSRGAN_s_y_model import Generator, Discriminator, ResidualBlock, UpsampleBLock  # 确保这里的模型类与你的代码匹配


# 自定义Dataset类
class ImageDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, transform=None):
        self.high_res_dir = high_res_dir  # 大尺寸图像文件夹路径
        self.low_res_dir = low_res_dir  # 小尺寸图像文件夹路径
        self.transform = transform
        self.high_res_images = os.listdir(high_res_dir)
        self.low_res_images = os.listdir(low_res_dir)

    def __len__(self):
        return len(self.high_res_images)

    def __getitem__(self, idx):
        high_res_image = Image.open(os.path.join(self.high_res_dir, self.high_res_images[idx])).convert('RGB')
        low_res_image = Image.open(os.path.join(self.low_res_dir, self.low_res_images[idx])).convert('RGB')

        if self.transform:
            high_res_image = self.transform(high_res_image)
            low_res_image = self.transform(low_res_image)

        return low_res_image, high_res_image


# 数据预处理，包括调整图像尺寸
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 这里的 (256, 256) 是你想要的尺寸，具体可以根据需要调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 数据加载器
train_dataset = ImageDataset(high_res_dir='data/train/high_res', low_res_dir='data/train/low_res', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 初始化模型
upscale_factor = 4  # 根据需要调整
generator = Generator(upscale_factor)
discriminator = Discriminator()

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)

# 损失函数
adversarial_loss = nn.BCELoss()
pixel_loss = nn.L1Loss()

# 优化器
lr = 1e-4
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))

# 标签初始化
real_label = 1
fake_label = 0

# 训练过程
num_epochs = 50
for epoch in range(num_epochs):
    for i, (low_res, high_res) in enumerate(train_loader):
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        # 训练判别器
        # 真实图像
        optimizer_D.zero_grad()
        real_pred = discriminator(high_res)
        real_loss = adversarial_loss(real_pred, torch.full_like(real_pred, real_label, device=device))

        # 生成图像
        fake_res = generator(low_res)
        fake_pred = discriminator(fake_res.detach())  # 只计算判别器损失
        fake_loss = adversarial_loss(fake_pred, torch.full_like(fake_pred, fake_label, device=device))

        # 总的判别器损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        fake_pred = discriminator(fake_res)  # 用生成器生成的假图像来计算生成器的损失
        g_loss = adversarial_loss(fake_pred, torch.full_like(fake_pred, real_label, device=device))

        # 像素损失（L1损失）
        pixel_loss_value = pixel_loss(fake_res, high_res)
        g_loss = g_loss + 100 * pixel_loss_value  # 强化像素级损失

        g_loss.backward()
        optimizer_G.step()

        # 打印训练进度
        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                  f"Pixel Loss: {pixel_loss_value.item():.4f}")

    # 每个epoch保存生成器的模型
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')

print("Training complete!")
