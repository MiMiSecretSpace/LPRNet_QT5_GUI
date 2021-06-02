import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal


class LPROpencvDtetction(QObject):
    finished = pyqtSignal()
    b_box = pyqtSignal(np.ndarray)

    def __init__(self, image):
        super(LPROpencvDtetction, self).__init__()
        self.image = image

    def detection(self):
        gray_scale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        x = cv2.Sobel(gray_scale, cv2.CV_16S, 1, 0)
        abs_x = cv2.convertScaleAbs(x)
        out = cv2.threshold(abs_x, 127, 255, cv2.THRESH_OTSU)
        kernel_x = np.ones((1, 3), np.uint8)
        kernel_y = np.ones((3, 1), np.uint8)
        dilation = cv2.dilate(out[1], kernel_x, iterations=2)
        erosion = cv2.erode(dilation, kernel_x, iterations=5)
        result_x = cv2.dilate(erosion, kernel_x, iterations=2)
        erosion = cv2.erode(result_x, kernel_y, iterations=1)
        result = cv2.dilate(erosion, kernel_y, iterations=2)
        contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 3)
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

        for bbox in bounding_boxes:
            [x, y, w, h] = bbox
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.b_box.emit(self.image)
        self.finished.emit()
