import cv2
img = cv2.imread("../material/test.jpg")
crop_img = img[10:50, 100:200]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)