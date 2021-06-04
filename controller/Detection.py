import time
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from model.LPRNet import LPRNet
from model.VehicleDetection import VehicleDetection
from model.LicensePlateRecognition import LicensePlateRecognition


class Detection(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(list)
    recognition_rate = pyqtSignal(float)

    def __init__(self, image: np.ndarray,
                 vehicle_detection_model: VehicleDetection,
                 lp_recognition: LicensePlateRecognition,
                 lprnet: LPRNet):
        super(Detection, self).__init__()
        self.image = image
        self.vehicle_detection_model = vehicle_detection_model
        self.lp_recognition = lp_recognition
        self.lprnet = lprnet

    def image_recognize(self):
        result = []
        plates = self.lp_recognition.scoring(self.image)
        for position in plates:
            plate = self.image[position[1]:position[3], position[0]:position[2]]
            if plate.shape[0] == 0 or plate.shape[1] == 0:
                return
            numbers = self.lprnet.test(plate)
            expression = self.lprnet.decode(numbers)
            result.append([position, expression])

        self.result.emit(result)
        self.recognition_rate.emit(time.process_time())
        self.finished.emit()
