import numpy as np
import cv2

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog


class Utils:
    @staticmethod
    def open_file_dialog(self, file_type, directory):
        file_name = QFileDialog.getOpenFileNames(self, 'Open file', directory, file_type)
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
    def bounding_box(self, image, left, top, right, bottom, vehicle) -> object:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 5)
        if vehicle is not None:
            cv2.putText(image, vehicle, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
