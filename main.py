import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from ui_main import Ui_MainWindow


class Constants:
    @staticmethod
    def open_file_dialog(self, file_type):
        file_name = QFileDialog.getOpenFileNames(self, 'Open file', '/home', file_type)
        if file_name[0]:
            return file_name[0]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('LPRNet GUI')
        self.ui.actionModel_name.triggered.connect(self.load_model)
        self.ui.actionLabel.triggered.connect(self.load_label)

    def load_model(self):
        ofd = Constants.open_file_dialog(self, '*.pb, *.pbtxt(*.pb *.pbtxt)')
        self.model = ofd

    def load_label(self):
        ofd = Constants.open_file_dialog(self, '*.txt(*.txt)')
        self.label = ofd

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
