import simulate_inco
import cv2
import numpy as np
import utils
import simulate_degrade
filename = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/AVG_lake.tif"
img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
img = np.array(img)
img2 = simulate_degrade.compress_img(img,2)
cv2.imwrite(r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/AVG_lake2.tif", img2)



