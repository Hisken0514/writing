import os
import shutil

def move_and_rename_images_by_fourth_digit(source_dir, dest_dir):
    # 檢查目標資料夾是否存在，如果不存在則創建
    for i in range(1,6):  # 建立0到9的資料夾
        folder_path = os.path.join(dest_dir, str(i))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # 初始化計數器字典，為每個目標資料夾維護一個計數器
    counters = {str(i): 1 for i in range(10)}
    
    # 遍歷源目錄中的所有文件
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg') and len(filename) >= 5:  # 僅處理長度足夠的圖片文件
            try:
                # 提取文件名中的第四個數字
                fourth_digit = filename[0]  # 文件名的第x個字符
                
                # 構造目標資料夾的路徑
                dest_folder = os.path.join(dest_dir, fourth_digit)
                
                # 構造新的文件名，格式為 "編號_原始文件名"
                new_filename = f"{counters[fourth_digit]:05d}_{filename}"
                
                # 構造完整的源文件和目標文件路徑
                source_file = os.path.join(source_dir, filename)
                dest_file = os.path.join(dest_folder, new_filename)
                
                # 移動並重命名文件
                shutil.move(source_file, dest_file)
                print(f"將 {filename} 重命名並移動到 {dest_folder} 資料夾，新的名稱為 {new_filename}")
                
                # 更新該資料夾的計數器
                counters[fourth_digit] += 1
                
            except IndexError:
                print(f"文件名格式錯誤，無法解析: {filename}")
            
source_dir = r"C:\Users\User\Documents\GitHub\CODE\mathedu\實驗備份\逐字resizew"  # 替換為你的源資料夾路徑
dest_dir = r'C:\Users\User\Documents\GitHub\CODE\mathedu\testclass'   # 替換為你的目標資料夾路徑

move_and_rename_images_by_fourth_digit(source_dir, dest_dir)
