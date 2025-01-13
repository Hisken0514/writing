import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    for label in range(3):  # 數字分類0~2
        dir_path = os.path.join(data_dir, str(label))
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=(50, 50))  #圖片大小50x50
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

data_dir = 'number'  # 數字目錄
x_data, y_data = load_data(data_dir)

# 檢查加載的圖像數據
print("原始圖像數據形狀:", x_data.shape)

# 處理數據
x_data = x_data.astype('float32') / 255
y_data = to_categorical(y_data)

# 檢查預處理後的圖像數據
print("預處理後的圖像數據形狀:", x_data.shape)

# 分割數據和訓練資料
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 改變數據形狀適應 RNN
x_train = x_train.reshape((x_train.shape[0], -1, 1))
x_test = x_test.reshape((x_test.shape[0], -1, 1))

# 建立模型
model = Sequential()
model.add(SimpleRNN(128, input_shape=(x_train.shape[1], 1), activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=50, verbose=2)

# 評估模型
scores = model.evaluate(x_test, y_test)
print('\n准确率:', scores[1])

# 預測
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# 顯示结果
def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

def show_image_labels_predictions(images, labels, predictions, start_id, num=3):
    plt.gcf().set_size_inches(12, 4)
    for i in range(0, num):
        ax = plt.subplot(1, num, 1 + i)
        # 顯示黑白照片
        ax.imshow(images[start_id].reshape(50, 50), cmap='binary')
        # AI預測结果
        if len(predictions) > 0:
            title = 'ai = ' + str(predictions[start_id])
            title += ' (o)' if predictions[start_id] == labels[start_id] else ' (x)'
            title += '\nlabel = ' + str(labels[start_id])
        else:
            title = 'label = ' + str(labels[start_id])

        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        start_id += 1
    plt.show()

# 顯示更多圖片的預測結果
num_images_to_show = 30  # 總共顯示 30 張圖片
batch_size = 10  # 每次顯示 10 張圖片

for start_id in range(0, num_images_to_show, batch_size):
    show_image_labels_predictions(x_test.reshape(-1, 50, 50), y_test.argmax(axis=1), predicted_classes, start_id=start_id, num=batch_size)
