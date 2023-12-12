import pydicom
import numpy as np
import cv2


path = 'output/vis/A_H07_N_9999999999_ST01_L_AXL_0295.dcm'
dcm = pydicom.dcmread(path)
dcm_img = dcm.pixel_array
dcm_img = dcm_img.astype(float)
# Rescaling grey scale between 0-255
dcm_img_scaled = (np.maximum(dcm_img, 0) / dcm_img.max()) * 255
# Convert to uint
dcm_img_scaled = np.uint8(dcm_img_scaled)

img = cv2.cvtColor(dcm_img_scaled, cv2.COLOR_GRAY2BGR)
# xyxy = [200, 220, 270, 265]
xyxy = [0, 0, 20, 26]

x1,y1,x2,y2 = xyxy
croped_img = img[y1:y2,x1:x2]
area = abs((y2-y1)*(x2-x1))
percent = (croped_img > 0).sum() / 3 / area
print(percent)
print(percent > .5)



