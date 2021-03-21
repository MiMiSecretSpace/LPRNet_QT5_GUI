import cv2
import sys
import numpy as np

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ui_main import Ui_MainWindow
from LPRNet import LPRNet

CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"  # exclude I, O
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}


class Constants:
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('LPRNet GUI')
        self.ui.actionModel.triggered.connect(self.load_model)
        self.ui.Image_path.clicked.connect(self.load_image)
        self.model = None
        self.label = None

    def load_model(self):
        ofd = Constants.open_file_dialog(self, '*.pb, *.pbtxt(*.pb *.pbtxt)')
        if ofd:
            self.model = ofd

    def load_label(self):
        ofd = Constants.open_file_dialog(self, '*.txt(*.txt)')
        if ofd:
            self.label = ofd

    def load_image(self):
        ofd = Constants.open_file_dialog(self, '*.jpg, *.jpge(*.jpg *.jpge)')
        if ofd:
            self.ui.Image_path_text.setText(ofd[0])
            self.image = cv2.imread(ofd[0])
            image = cv2.resize(self.image, (800, 600))
            image = QtGui.QImage(image,
                                 image.shape[1],
                                 image.shape[0],
                                 image.strides[0],
                                 QtGui.QImage.Format_RGB888)
            image_frame = QtWidgets.QLabel()
            image_frame.setPixmap(QtGui.QPixmap.fromImage(image))
            Constants.clear_hbox(self, self.ui.horizontalLayout)
            self.ui.horizontalLayout.addWidget(image_frame)
            if self.model:
                self.recognize()
            else:
                msg = QMessageBox()
                msg.setWindowTitle("No model error")
                msg.setText("No model loaded !")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

    def recognize(self):
        image = cv2.resize(self.image, (94, 24))
        img_batch = np.expand_dims(image, axis=0)
        model = LPRNet(model_filepath=self.model[0])
        result = model.test(data=img_batch)

        decoded_labels = []
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
