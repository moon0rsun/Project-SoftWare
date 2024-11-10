import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt


# 自定义数据集
class SARDataset(Dataset):
    def __init__(self, img_folder1, img_folder2, label_folder, transform=None):
        self.img_folder1 = img_folder1
        self.img_folder2 = img_folder2
        self.label_folder = label_folder
        self.transform = transform
        self.img1_list = sorted(os.listdir(img_folder1))
        self.img2_list = sorted(os.listdir(img_folder2))
        self.label_list = sorted(os.listdir(label_folder))

    def __len__(self):
        return len(self.img1_list)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.img_folder1, self.img1_list[idx])
        img2_path = os.path.join(self.img_folder2, self.img2_list[idx])
        label_path = os.path.join(self.label_folder, self.label_list[idx])

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            label = self.transform(label)

        return img1, img2, label


# CNN 模型
class ChangeDetectionCNN(nn.Module):
    def __init__(self):
        super(ChangeDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)  # 输出单通道
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 上采样到原尺寸

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), dim=1)  # 合并两个图像
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))  # 生成二值变化图
        x = self.upsample(x)  # 上采样回到标签的大小
        return x


# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, output_folder='./output'):
    model.train()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (img1, img2, label) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(img1, img2)
            outputs = outputs.squeeze(1)  # 移除通道维度
            label = label.squeeze(1)  # 标签也需要移除通道维度

            loss = criterion(outputs, label.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 保存输出图像（每100个batch保存一次）
            if i % 100 == 0:
                output_img_path = os.path.join(output_folder, f"epoch_{epoch + 1}_batch_{i}.png")
                save_output_image(outputs[0], output_img_path)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


def save_output_image(output_tensor, output_path):
    # 反转 Tensor 的归一化并保存为图像
    output_image = output_tensor.cpu().detach().numpy()
    plt.imsave(output_path, output_image, cmap='gray')


# 加载数据集和模型
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
train_dataset = SARDataset('../data/train/img1', '../data/train/img2', '../data/train/labels', transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = ChangeDetectionCNN()
criterion = nn.BCELoss()  # 使用二进制交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10, output_folder='./output')
