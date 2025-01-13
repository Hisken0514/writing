import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 显示单张图像
def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

# 显示图像、标签和预测结果
def show_image_labels_predictions(images, labels, predictions, start_id, num=10):
    plt.gcf().set_size_inches(12, 14)
    if num > 25: num = 25

    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        # 显示黑白照片
        ax.imshow(images[start_id], cmap='binary')
        # AI预测结果
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

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 建立模型
model = Sequential()
model.add(Dense(units=1024, input_dim=784, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=100, verbose=2)

# 評估模型
scores = model.evaluate(x_test, y_test)
print('\n准确率:', scores[1])

# 預測
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# 顯示结果
show_image_labels_predictions(x_test.reshape(-1, 28, 28), y_test.argmax(axis=1), predicted_classes, start_id=0, num=10)
