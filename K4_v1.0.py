import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE

# 确保 TensorFlow 使用 GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU")
else:
    print("No GPU found, using CPU")

# 加载数据的函数
def load_data(data_dir):
    images = []
    labels = []
    for label in range(5):  # 假设有 5 类数据
        dir_path = os.path.join(data_dir, str(label))
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=(30, 30))  # 调整图片大小
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# 数据目录
data_dir = 'writing'  # 替换为你的数据目录
x_data, y_data = load_data(data_dir)

# 数据处理：归一化
x_data = x_data.astype('float32') / 255
y_data = to_categorical(y_data)  # 转换标签为 one-hot 编码

# 使用 K4 进行训练和评估
kf = KFold(n_splits=4, shuffle=True)
accuracies = []

# 设置早停回调
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for train_index, test_index in kf.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    # 建立模型
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=2, validation_data=(x_test, y_test))
    
    # 评估模型
    scores = model.evaluate(x_test, y_test, verbose=0)
    accuracies.append(scores[1])

# 显示 K4 结果
print('\n平均準確率:', np.mean(accuracies))

# 在整个数据集上重新训练模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(x_data, y_data, epochs=10, batch_size=30, verbose=2, validation_split=0.1)

# 在所有数据上进行预测
predictions = model.predict(x_data)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_data, axis=1)

# 将数据从 50x50x1 展平为 2500 维（用于降维）
x_flatten = x_data.reshape(x_data.shape[0], -1)

# 使用 t-SNE 将数据降到 2 维
tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
x_tsne = tsne.fit_transform(x_flatten)

# 绘制 2D 散点图，展示 t-SNE 的降维效果，并区分模型预测的分类结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=predicted_labels, cmap='viridis', s=10)
plt.colorbar(scatter, label='Predicted Class')
plt.title('2D t-SNE of Image Data After Model Prediction')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# 混淆矩陣
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 顯示混淆矩陣
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 评估模型在所有数据上的表现
scores = model.evaluate(x_data, y_data, verbose=0)
print('\n整個數據的準確率:', scores[1])
