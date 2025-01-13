import os
from PIL import Image
import re

# 原始資料夾路徑
input_folder = r"C:\Users\User\Documents\GitHub\CODE\mathedu\實驗備份\逐字\"
# 新的資料夾路徑
output_folder = r"C:\Users\User\Documents\GitHub\CODE\mathedu\testout"

# 創建新的資料夾
os.makedirs(output_folder, exist_ok=True)

# 設定目標大小
target_size = (600, 600)

# 檢查 output_folder 中已經存在的圖片，以獲取最大的現有編號
existing_files = [f for f in os.listdir(output_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
# 使用正則表達式提取檔案名中的數字
number_pattern = re.compile(r'(\d{2})_(\d{4})')
existing_numbers = [int(match.group(2)) for f in existing_files if (match := number_pattern.match(f))]
image_counter = max(existing_numbers, default=0) + 1  # 從最大的現有編號 + 1 開始

# 遍歷每個子資料夾和圖片
for root, dirs, files in os.walk(input_folder):
    for file in files:
        # 確保是圖片格式
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 提取檔案名稱中的第四個和第五個數字
            if len(file) >= 5:  # 確保檔名至少有五個字符
                digit1 = file[3]
                digit2 = file[4]
            else:
                print(f"檔名過短，無法提取數字，跳過文件：{file}")
                continue

            # 生成新的圖片名稱（字首為兩個數字，接續編號）
            new_image_name = f"{digit1}{digit2}_{image_counter:04d}.jpg"
            image_counter += 1

            img_path = os.path.join(root, file)
            # 打開圖片
            with Image.open(img_path) as img:
                # 調整圖片大小
                img.thumbnail(target_size, Image.LANCZOS)
                # 創建新圖片並填充背景
                new_img = Image.new("RGB", target_size, (160, 150, 150))
                new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                                     (target_size[1] - img.size[1]) // 2))

                # 構建新的圖片保存路徑
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # 保存新圖片
                new_img.save(os.path.join(output_subfolder, new_image_name))

print("所有圖片已按指定格式命名並保存到新的資料夾中！")
