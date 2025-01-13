import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

x = 30 #調整圖片的size

def load_data(data_dir):
    images = []
    labels = []
    for label in range(5):  #  0-4 的分類
        dir_path = os.path.join(data_dir, str(label))
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=(x, x)) 
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

data_dir = 'writing'  #圖片檔案區
x_data, y_data = load_data(data_dir)

# 檢查加載的圖像數據
print("原始圖像數據形狀:", x_data.shape)

x_data = x_data.astype('float32') / 255
y_data = to_categorical(y_data)

print("預處理後的圖像數據形狀:", x_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(x, x, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.3, epochs=200, batch_size=10, verbose=1)

model_path = 'my_model.h5'
model.save(model_path)
print(f'Model saved to {model_path}')

scores = model.evaluate(x_test, y_test)
print('\n準確率:', scores[1])

predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

def show_image_labels_predictions(images, labels, predictions, start_id, num=30, images_per_row=10):
    num_rows = (num // images_per_row) + int(num % images_per_row != 0)  # 計算總行數
    plt.figure(figsize=(images_per_row * 2, num_rows * 2))  # 調整畫布大小
    
    for i in range(0, num):
        ax = plt.subplot(num_rows, images_per_row, 1 + i)
        ax.imshow(images[start_id].reshape(x, x), cmap='binary')
        
        if len(predictions) > 0:
            title = 'ai = ' + str(predictions[start_id])
            title += ' (o)' if predictions[start_id] == labels[start_id] else ' (x)'
            title += '\nlabel = ' + str(labels[start_id])
        else:
            title = 'label = ' + str(labels[start_id])

        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        start_id += 1
        
    plt.tight_layout()  # 調整子圖之間的佈局，避免重疊
    plt.show()

# 顯示所有圖片的預測結果
num_images_to_show = 30  # 顯示 30 張圖片
show_image_labels_predictions(x_test, y_test.argmax(axis=1), predicted_classes, start_id=0, num=num_images_to_show, images_per_row=10)


loaded_model = tf.keras.models.load_model('my_model.h5')
loaded_scores = loaded_model.evaluate(x_test, y_test)
print('\n模型準確率:', loaded_scores[1])