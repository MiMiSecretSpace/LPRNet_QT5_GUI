import cv2
import sys
import numpy as np
import time

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, QObject, QThread, pyqtSignal

from ui_main import Ui_MainWindow
from LPRNet import LPRNet
from ObjectDetection import ObjectDetection
from opencv_license_plate_detection import LPRdtetction

CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"  # exclude I, O
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}

VEHICLE = ["person", "car", "motorcycle", "bus", "truck"]
VEHICLE_ID = [0, 2, 3, 5, 7]


class Utils:
    @staticmethod
    def open_file_dialog(self, file_type):
        file_name = QFileDialog.getOpenFileNames(self, 'Open file', '/home', file_type)
        if file_name[0]:
            return file_name[0]

    @staticmethod
    def clear_hbox(self, layout=False):
        if not layout:
            layout = self.ui.horizontalLayout
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_hbox(item.layout())

    @staticmethod
    def show_image(self, image: np.ndarray, label: QtWidgets.QLabel):
        resize_image = cv2.resize(image, (label.width(), label.height()))
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        display_image = QtGui.QImage(resize_image,
                                     resize_image.shape[1],
                                     resize_image.shape[0],
                                     resize_image.strides[0],
                                     QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(display_image))

    @staticmethod
    def set_text(self, label: QtWidgets.QLabel, text: str):
        label.setText(text)

    @staticmethod
    def bounding_box(self, image, left, top, right, bottom, label_id):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 5)
        if label_id is not None:
            cv2.putText(image, VEHICLE[VEHICLE_ID.index(label_id)], (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


class Detection(QObject):
    finished = pyqtSignal()
    bounding_box = pyqtSignal(list)
    recognition_rate = pyqtSignal(float)

    def __init__(self, image, label, object_detection_model, lprnet_model):
        super(Detection, self).__init__()
        self.image = image
        self.label = label
        self.object_detection_model = object_detection_model
        self.lprnet_model = lprnet_model
        self.lpr_detection = LPRdtetction(image)

    def image_recognize(self):
        img_height, img_width, img_channels = self.image.shape
        crop_images = []

        #result[2] = result[0] + result[2]
        #result[3] = result[1] + result[3]
        #result = self.lpr_detection.detection()
        #crop_images = result

        if self.object_detection_model:
            result = self.object_detection_model.test(data=self.image)
            for i in range(10):
                score = result[2][0][i]
                classes = result[1][0][i]
                if classes == 0 or classes == 2 or classes == 3 or classes == 5 or classes == 7:
                    if score > 0.5:
                        top = abs(int(result[0][0][i][0] * img_height))
                        left = abs(int(result[0][0][i][1] * img_width))
                        bottom = abs(int(result[0][0][i][2] * img_height))
                        right = abs(int(result[0][0][i][3] * img_width))
                        crop_images.append([left, top, right, bottom, classes])
                        # crop_images.append(self.image[top:bottom, left:right])
                        # cv2.rectangle(self.image, (left, top), (right, bottom), (0, 255, 0), 5)

        # Utils.show_image(self, self.image, self.label)

        if self.lprnet_model:
            for i in range(len(crop_images)):
                result = self.lprnet_model.test(crop_images[i])
                for item in result:
                    # print(item)
                    expression = ['' if i == -1 else DECODE_DICT[i] for i in item]
                    expression = ''.join(expression)

        self.bounding_box.emit(crop_images)
        self.recognition_rate.emit(time.process_time())
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.image = np.ndarray
        self.lprnet_model = None
        self.object_detection_model = None
        self.label = None
        self.cap = cv2.VideoCapture
        self.fps = 0
        self.timer = QTimer(self)
        self.thread = QThread()
        self.crop_images = []
        self.last_time = 0
        self.init_slots()

    def init_slots(self):
        self.ui.actionLPRNet.triggered.connect(self.load_lprnet_model)
        self.ui.actionObject_Detection.triggered.connect(self.load_object_detection_model)
        self.ui.Image_path.clicked.connect(self.load_image)
        self.ui.Video_path.clicked.connect(self.load_video)
        self.ui.play_buttom.clicked.connect(self.play_pause)
        self.timer.timeout.connect(self.timer_tick)

    def show_recognition_rate(self, t):
        t = int(t * 1000)
        self.ui.recognition_rate.setText(str(t - self.last_time) + ' ms')
        self.last_time = t

    def pre_bounding(self, crop_images):
        self.crop_images = crop_images
        for i in range(len(self.crop_images)):
            Utils.bounding_box(self, self.image, self.crop_images[i][0],
                               self.crop_images[i][1], self.crop_images[i][2],
                               self.crop_images[i][3], self.crop_images[i][4])

    def play_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.ui.play_buttom.setText('Play')
        else:
            self.timer.start(1000 / self.fps)
            self.ui.play_buttom.setText('Pause')

    def timer_tick(self):
        ret, self.image = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, self.image = self.cap.read()

        if not self.thread.isRunning():
            self.thread = QThread()
            self.worker = Detection(self.image, self.ui.video_holder, self.object_detection_model, self.lprnet_model)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.image_recognize)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            # self.thread.finished.connect(self.thread.deleteLater)
            self.worker.bounding_box.connect(self.pre_bounding)
            self.worker.recognition_rate.connect(self.show_recognition_rate)
            self.thread.start()
        else:
            for i in range(len(self.crop_images)):
                Utils.bounding_box(self, self.image, self.crop_images[i][0],
                                   self.crop_images[i][1], self.crop_images[i][2],
                                   self.crop_images[i][3], self.crop_images[i][4])

        Utils.show_image(self, self.image, self.ui.video_holder)

    def load_lprnet_model(self):
        ofd = Utils.open_file_dialog(self, '*.pb, *.pbtxt(*.pb *.pbtxt)')
        if ofd:
            self.lprnet_model = LPRNet(model_filepath=ofd[0])
            self.ui.statusbar.showMessage('LPRNet success loaded !', 2000)

    def load_object_detection_model(self):
        ofd = Utils.open_file_dialog(self, '*.tflite(*.tflite)')
        if ofd:
            self.object_detection_model = ObjectDetection(model_filepath=ofd[0])
            self.ui.statusbar.showMessage('Object detection model success loaded !', 2000)

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

        self.thread = QThread()
        self.worker = Detection(self.image, self.ui.image_holder, self.object_detection_model, self.lprnet_model)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.image_recognize)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
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
