import cv2
import sys
import numpy as np

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer

from ui_main import Ui_MainWindow
from LPRNet import LPRNet
from ObjectDetection import ObjectDetection

CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"  # exclude I, O
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}


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
    def show_image(self, image, label):
        resize_image = cv2.resize(image, (label.width(), label.height()))
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        display_image = QtGui.QImage(resize_image,
                                     resize_image.shape[1],
                                     resize_image.shape[0],
                                     resize_image.strides[0],
                                     QtGui.QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(display_image))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.image = ''
        self.ui.setupUi(self)
        self.setWindowTitle('LPRNet GUI')
        self.ui.actionLPRNet.triggered.connect(self.load_lprnet_model)
        self.ui.actionObject_Detection.triggered.connect(self.load_object_detection_model)
        self.ui.Image_path.clicked.connect(self.load_image)
        self.ui.Video_path.clicked.connect(self.load_video)
        self.lprnet_model = None
        self.object_detection_model = None
        self.label = None
        self.cap = ''
        self.fps = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timer_tick)
        self.ui.play_buttom.clicked.connect(self.play_pause)

    def play_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.ui.play_buttom.setText('Play')
        else:
            self.timer.start(1000/self.fps)
            self.ui.play_buttom.setText('Pause')

    def timer_tick(self):
        ret, self.image = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, self.image = self.cap.read()
        self.img_height, self.img_width, self.img_channles = self.image.shape
        self.image_recognize(self.ui.video_holder)

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
        self.img_height, self.img_width, self.img_channles = self.image.shape

        if self.object_detection_model or self.lprnet_model:
            self.image_recognize(self.ui.image_holder)
        else:
            msg = QMessageBox()
            msg.setWindowTitle("No model error")
            msg.setText("No model loaded !")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def load_video(self):
        ofd = Utils.open_file_dialog(self, '*.mp4, *.avi(*.mp4 *.avi)')
        if not ofd:
            return
        self.ui.video_path_text.setText(ofd[0])
        self.cap = cv2.VideoCapture(ofd[0])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def image_recognize(self, label):
        crop_images = []
        if self.object_detection_model:
            result = self.object_detection_model.test(data=self.image)
            for i in range(10):
                score = result[2][0][i]
                if score > 0.001:
                    top = abs(int(result[0][0][i][0] * self.img_height))
                    left = abs(int(result[0][0][i][1] * self.img_width))
                    bottom = abs(int(result[0][0][i][2] * self.img_height))
                    right = abs(int(result[0][0][i][3] * self.img_width))
                    crop_images.append(self.image[top:bottom, left:right])
                    cv2.rectangle(self.image, (left, top), (right, bottom), (0, 255, 0), 5)

        Utils.show_image(self, self.image, label)

        if self.lprnet_model:
            for i in range(len(crop_images)):
                result = self.lprnet_model.test(crop_images[i])
                for item in result:
                    # print(item)
                    expression = ['' if i == -1 else DECODE_DICT[i] for i in item]
                    expression = ''.join(expression)

                self.ui.textEdit.setText(expression)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
