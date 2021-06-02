import cv2
import sys
import numpy as np
import time

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer, QObject, QThread, pyqtSignal

from view.ui_main import Ui_MainWindow
from model.LPRNet import LPRNet
from model.VehicleDetection import VehicleDetection
from model.LicensePlateRecognition import LicensePlateRecognition
from conrtoller.Utils import Utils

CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"  # exclude I, O
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}


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
            numbers = self.lprnet.test(plate)
            for item in numbers:
                # print(item)
                expression = ['' if i == -1 else DECODE_DICT[i] for i in item]
                expression = ''.join(expression)
            result.append([position, expression])
        '''
        
        '''
        '''
        if self.lprnet_model:
            for i in range(len(crop_images)):
                result = self.lprnet_model.test(crop_images[i])
                for item in result:
                    # print(item)
                    expression = ['' if i == -1 else DECODE_DICT[i] for i in item]
                    expression = ''.join(expression)

        '''
        self.result.emit(result)
        self.recognition_rate.emit(time.process_time())
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.image = np.ndarray
        self.lprnet_model = LPRNet('../material/LPRNet.pb')
        self.vehicle_detection_model = VehicleDetection('../material/ObjectModel.tflite')
        self.lp_recognition_model = LicensePlateRecognition('../material/LicensePlateRecognition.tflite')
        self.label = None
        self.cap = cv2.VideoCapture
        self.fps = 0
        self.timer = QTimer(self)
        self.thread = QThread()
        self.plates = None
        self.last_time = 0
        self.init_slots()

    def init_slots(self):
        self.ui.actionLPRNet.triggered.connect(self.set_lprnet_model)
        self.ui.actionObject_Detection.triggered.connect(self.set_vehicle_detection_model)
        self.ui.actionLicense_Plate_Recognition.triggered.connect(self.set_license_plate_recognition_model)
        self.ui.Image_path.clicked.connect(self.load_image)
        self.ui.Video_path.clicked.connect(self.load_video)
        self.ui.play_buttom.clicked.connect(self.play_pause)
        self.timer.timeout.connect(self.timer_tick)

    def show_recognition_rate(self, t):
        t = int(t * 1000)
        self.ui.recognition_rate.setText(str(t - self.last_time) + ' ms')
        self.last_time = t

    def play_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.ui.play_buttom.setText('Play')
        else:
            self.timer.start(1000 / self.fps)
            self.ui.play_buttom.setText('Pause')

    def set_plate(self, plates):
        self.plates = plates
        '''
        Utils.bounding_box(self, self.image, self.plates[0],
                           self.plates[1], self.plates[2],
                           self.plates[3], None)
        '''
        for plate in self.plates:
            Utils.bounding_box(self, self.image, plate[0][0],
                               plate[0][1], plate[0][2],
                               plate[0][3], plate[1])
        #Utils.show_image(self, self.image, self.ui.image_holder)

    def detection_threading(self, use_image):
        if not self.thread.isRunning():
            self.thread = QThread()
            self.worker = Detection(self.image, self.vehicle_detection_model,
                                    self.lp_recognition_model, self.lprnet_model)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.image_recognize)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            # self.thread.finished.connect(self.thread.deleteLater)
            self.worker.result.connect(self.set_plate)
            self.worker.recognition_rate.connect(self.show_recognition_rate)
            if use_image:
                self.worker.finished.connect(lambda: Utils.show_image(self, self.image, self.ui.image_holder))
            self.thread.start()
        else:
            if self.plates is not None:
                self.set_plate(self.plates)

    def timer_tick(self):
        ret, self.image = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, self.image = self.cap.read()
        self.detection_threading(use_image=False)
        Utils.show_image(self, self.image, self.ui.video_holder)

    def set_lprnet_model(self):
        ofd = Utils.open_file_dialog(self, '*.pb, *.pbtxt(*.pb *.pbtxt)')
        if ofd:
            self.lprnet_model = LPRNet(model_filepath=ofd[0])
            self.ui.statusbar.showMessage('LPRNet success loaded !', 2000)

    def set_vehicle_detection_model(self):
        ofd = Utils.open_file_dialog(self, '*.tflite(*.tflite)')
        if ofd:
            self.vehicle_detection_model = VehicleDetection(model_filepath=ofd[0])
            self.ui.statusbar.showMessage('Vehicle detection model success loaded !', 2000)

    def set_license_plate_recognition_model(self):
        ofd = Utils.open_file_dialog(self, '*.tflite(*.tflite)')
        if ofd:
            self.lp_recognition_model = LicensePlateRecognition(model_filepath=ofd[0])
            self.ui.statusbar.showMessage('License plate recognition model success loaded !', 2000)

    def load_label(self):
        ofd = Utils.open_file_dialog(self, '*.txt(*.txt)')
        if ofd:
            self.label = ofd

    def load_image(self):
        ofd = Utils.open_file_dialog(self, '*.jpg, *.jpge(*.jpg *.jpge)')
        if not ofd:
            return
        self.ui.Image_path_text.setText(ofd[0])
        self.image = cv2.imread(ofd[0])
        self.detection_threading(use_image=True)
        while self.thread.isFinished():
            Utils.show_image(self, self.image, self.ui.image_holder)

        '''
        if self.object_detection_model or self.lprnet_model:
            self.image_recognize(self.ui.image_holder)
        else:
            msg = QMessageBox()
            msg.setWindowTitle("No model error")
            msg.setText("No model loaded !")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        '''

    def load_video(self):
        ofd = Utils.open_file_dialog(self, '*.mp4, *.avi(*.mp4 *.avi)')
        if not ofd:
            return
        self.ui.video_path_text.setText(ofd[0])
        self.cap = cv2.VideoCapture(ofd[0])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
