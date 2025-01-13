import os   #LENET
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號無法顯示的問題

# 設定設備為 GPU 或 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 圖片大小
width, height = 200, 200

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
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  
        # 第二層卷積層，輸入通道為6，輸出通道為16，卷積核大小為 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  
        # 計算展平後的大小
        self.flattened_size = self._get_flattened_size()
        # 全連接層
        self.fc1 = nn.Linear(self.flattened_size, 120)  # 第一個全連接層
        self.fc2 = nn.Linear(120, 84)  # 第二個全連接層
        self.fc3 = nn.Linear(84, 2)   # 輸出層，將輸出大小設為5，對應於你的5個類別
    def _get_flattened_size(self):
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
        x = self.fc3(x)                           
        return x

# 載入資料集並劃分訓練集與測試集
data_dir = r'C:\Users\User\Documents\GitHub\CODE\mathedu\testclass'
dataset = CustomDataset(data_dir, transform)

# 分割數據集成訓練和測試集 (例如80%訓練, 20%測試)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、損失函數和優化器
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

# 訓練模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

# 評估模型
model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 混淆矩陣
conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(1, 3))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 保存模型
torch.save(model.state_dict(), 'exmodel_vol01.pth')
