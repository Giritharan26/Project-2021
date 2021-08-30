import cv2
import numpy as np
import csv

data = []
import glob
cv_img = []
for img in glob.glob("C:\\Users\\Win 7\\Desktop\\Machine learing\\test\\*.jpg"):
    n= cv2.imread(img)
    print(img)
    cv_img.append(n)
    dim = (224,224)
    resized = cv2.resize(n, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(img,resized)



