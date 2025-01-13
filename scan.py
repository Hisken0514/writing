import cv2 as cv
import numpy as np

image_path = r"C:\Users\User\Documents\GitHub\CODE\mathedu\實驗備份\逐字resizew- 複製\53_1496.jpg"

image = cv.imread(image_path)
img_resized = cv.resize(image, (600,600))  #調整圖片縮放大小

gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

cv.imshow('Result', gray)
cv.waitKey(0)
