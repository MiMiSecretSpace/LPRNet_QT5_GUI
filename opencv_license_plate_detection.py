import cv2
import numpy as np

from google.colab.patches import cv2_imshow


carplate_img = cv2.imread("a.jpg")
grayScale = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2GRAY)
x = cv2.Sobel(grayScale, cv2.CV_16S, 1, 0)
absX = cv2.convertScaleAbs(x)
out = cv2.threshold(absX, 127, 255, cv2.THRESH_OTSU)
kernelX = np.ones((1,3), np.uint8)
kernelY = np.ones((3,1), np.uint8)
dilation = cv2.dilate(out[1], kernelX, iterations = 2)
erosion = cv2.erode(dilation, kernelX, iterations = 5)
resultX = cv2.dilate(erosion, kernelX, iterations = 2)
erosion = cv2.erode(resultX, kernelY, iterations = 1)
result = cv2.dilate(erosion, kernelY, iterations = 2)
contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(result, contours, -1, (0, 0, 255), 3)
bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
 
for bbox in bounding_boxes:
     [x , y, w, h] = bbox
     cv2.rectangle(carplate_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2_imshow(carplate_img)
