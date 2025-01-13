from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 1. 載入已保存的模型
model = load_model('modelex1.h5')
# 3. 針對單張圖片進行預測
img_path = 't12.jpg'  # 替換為你圖片的路徑

# 2. 載入和預處理單張圖片
def preprocess_image(img_path):
    img = load_img(img_path, color_mode='grayscale', target_size=(100, 30))  # 調整大小為 (100, 30)
    img_array = img_to_array(img)  # 轉換成數值型別
    img_array = img_array.astype('float32') / 255  # 歸一化
    img_array = np.expand_dims(img_array, axis=0)  # 增加一個維度，模擬批量處理 [1, 100, 30, 1]
    return img_array



processed_image = preprocess_image(img_path)

# 進行預測
prediction = model.predict(processed_image)

# 4. 解析預測結果
predicted_label = np.argmax(prediction, axis=1) + 1  # 假設標籤範圍是 1~5
print(f"Predicted label: {predicted_label[0]}")
