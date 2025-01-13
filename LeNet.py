import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號無法顯示的問題
# 設定設備為 GPU 或 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 圖片大小
width, height = 30, 30

# 資料轉換設定（包括數據增強技術）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(15),  # 隨機旋轉 -15 到 15 度
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.RandomResizedCrop((width, height), scale=(0.8, 1.0)),  # 隨機裁剪並調整大小
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 調整亮度和對比度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 將影像數據正規化到 [-1, 1] 的範圍內
])

# 自定義資料集類
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label in range(1, 6):  # 假設標籤為1~5
            dir_path = os.path.join(data_dir, str(label))
            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)
                img = Image.open(img_path).convert('RGB')  # 使用 PIL 開啟圖片並轉換為 RGB 模式
                if self.transform:
                    img = self.transform(img)
                self.images.append(img)
                self.labels.append(label - 1)  # 將標籤從1~5變為0~4

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# LeNet-5 模型類
class LeNet5Model(nn.Module):
    def __init__(self):
        super(LeNet5Model, self).__init__()
        # 第一層卷積層，輸入通道為1，輸出通道為6，卷積核大小為 5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  
        # 第二層卷積層，輸入通道為6，輸出通道為16，卷積核大小為 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  
        
        # 計算展平後的大小
        self.flattened_size = self._get_flattened_size()

        # 全連接層
        self.fc1 = nn.Linear(self.flattened_size, 120)  # 第一個全連接層
        self.fc2 = nn.Linear(120, 84)  # 第二個全連接層
        self.fc3 = nn.Linear(84, 5)   # 輸出層，將輸出大小設為5，對應於你的5個類別

    def _get_flattened_size(self):
        # 使用一個假設的輸入來計算展平大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, width, height)
            x = F.avg_pool2d(F.relu(self.conv1(dummy_input)), 2)
            x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
            return x.view(-1).shape[0]

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)  # Conv1 -> ReLU -> Pool
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)  # Conv2 -> ReLU -> Pool
        x = x.view(-1, self.flattened_size)         # 展平
        
        x = F.relu(self.fc1(x))                     # 全連接層1
        x = F.relu(self.fc2(x))                     # 全連接層2
        x = self.fc3(x)                             # 輸出層
        return x




# 載入資料集
data_dir = r'C:\Users\User\Documents\GitHub\CODE\mathedu\testclass'  # 替換為你的資料目錄
dataset = CustomDataset(data_dir, transform)

# 使用 LeNet-5 模型
model = LeNet5Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

# 訓練和驗證損失的列表
train_losses = []
val_losses = []

# 執行 LOOCV 訓練
all_predictions = []
all_labels = []

# tqdm to track overall LOOCV progress
for idx in tqdm(range(len(dataset)), desc="LOOCV Progress"):
    # 切分出訓練集和測試集
    train_indices = list(range(len(dataset)))
    train_indices.pop(idx)  # 移除當前索引作為訓練集
    test_indices = [idx]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    # 訓練模型
    model.train()
    num_epochs = 10  # 可以根據需要調整 epoch 數量
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # 每個 epoch 的損失
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向傳播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向傳播和優化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # 計算每個 epoch 的平均訓練損失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 驗證模型
        model.eval()
        val_loss = 0.0  # 驗證損失
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # 計算驗證損失
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

    # 測試模型
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted.item())
            all_labels.append(labels.item())

# 混淆矩陣
conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(1,6))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 繪製訓練損失和驗證損失
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='訓練損失', color='blue')
plt.plot(val_losses, label='驗證損失', color='orange')
plt.title('訓練損失和驗證損失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# t-SNE 降維
all_images = np.array([dataset[i][0].numpy().flatten() for i in range(len(dataset))])
tsne = TSNE(n_components=2, perplexity=5, n_iter=800)
x_tsne = tsne.fit_transform(all_images)

# 繪製 t-SNE 結果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=all_predictions, cmap='viridis', s=10)
plt.colorbar(scatter, label='Predicted Class')
plt.title('2D t-SNE of Image Data After Model Prediction')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
torch.save(model.state_dict(), 'exmodel_vol01.pth')
