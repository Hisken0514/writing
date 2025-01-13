import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號無法顯示的問題

# 設定設備為 GPU 或 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 圖片大小
width, height = 360,200

# 資料轉換設定（包括數據增強技術）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((width, height), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 自定義資料集類
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label in range(1, 3):  # 假設標籤為1~5
            dir_path = os.path.join(data_dir, str(label))
            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                self.images.append(img)
                self.labels.append(label - 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# CNN 模型類
class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Conv Layer 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv Layer 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv Layer 3
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 自適應池化至 4x4 特徵圖大小
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 計算展平後的大小
        self.flattened_size = 256 * 4 * 4

        # 全連接層
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 卷積層
        x = x.view(-1, self.flattened_size)  # 展平
        x = self.fc_layers(x)  # 全連接層
        return x

# 載入資料集
data_dir = r'C:\Users\User\Documents\GitHub\CODE\mathedu\testclass'
dataset = CustomDataset(data_dir, transform)

# K-fold cross-validation (K=10)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
num_epochs = 10

all_fold_predictions = []
all_fold_labels = []

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{kf.n_splits}")

    # 切分訓練集與測試集
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    # 初始化模型、損失函數和優化器
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # 訓練模型
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Fold {fold+1}, Loss: {train_loss / len(train_loader):.4f}")

    # 評估模型
    model.eval()
    fold_predictions = []
    fold_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            fold_predictions.extend(predicted.cpu().numpy())
            fold_labels.extend(labels.cpu().numpy())

    # 保存每個fold的結果
    all_fold_predictions.extend(fold_predictions)
    all_fold_labels.extend(fold_labels)

# 混淆矩陣
conf_matrix = confusion_matrix(all_fold_labels, all_fold_predictions)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(1, 3))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 保存最終模型
torch.save(model.state_dict(), 'exmodel_vol01_k10.pth')
