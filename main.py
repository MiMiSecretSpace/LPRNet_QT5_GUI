import cv2
import sys
import numpy as np

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox

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
        display_image = QtGui.QImage(resize_image,
                                     resize_image.shape[1],
                                     resize_image.shape[0],
                                     resize_image.strides[0],
                                     QtGui.QImage.Format_BGR888)
        label.setPixmap(QtGui.QPixmap.fromImage(display_image))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('LPRNet GUI')
        self.ui.actionLPRNet.triggered.connect(self.load_lprnet_model)
        self.ui.actionObject_Detection.triggered.connect(self.load_object_detection_model)
        self.ui.Image_path.clicked.connect(self.load_image)
        self.ui.Video_path.clicked.connect(self.load_video)
        self.lprnet_model = None
        self.object_detection_model = None
        self.label = None

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
        if ofd:
            self.ui.Image_path_text.setText(ofd[0])
            self.image = cv2.imread(ofd[0])
            self.img_height, self.img_width, self.img_channles = self.image.shape

            if self.object_detection_model:
                self.image_recognize()
            else:
                msg = QMessageBox()
                msg.setWindowTitle("No model error")
                msg.setText("No model loaded !")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

    def load_video(self):
        ofd = Utils.open_file_dialog(self, '*.mp4, *.avi(*.mp4 *.avi)')
        if ofd:
            self.ui.Image_path_text_2.setText(ofd[0])

    def image_recognize(self):
        result = self.object_detection_model.test(data=self.image)
        for i in range(len(result[2])):
            score = result[2][0][i]
            if score > 0.05:
                top = int(result[0][0][i][0] * self.img_height)
                left = int(result[0][0][i][1] * self.img_width)
                bottom = int(result[0][0][i][2] * self.img_height)
                right = int(result[0][0][i][3] * self.img_width)
                cv2.rectangle(self.image, (left, top), (right, bottom), (0, 255, 0), 5)
        Utils.show_image(self, self.image, self.ui.image_holder)
        #crop_image = self.image[top:bottom, left:right]
        #result = self.lprnet_model.test(crop_image)
        #for item in result:
        #    # print(item)
        #    expression = ['' if i == -1 else DECODE_DICT[i] for i in item]
        #    expression = ''.join(expression)
        #
        #self.ui.textEdit.setText(expression)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
