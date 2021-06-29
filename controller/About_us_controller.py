from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog

from view.About_us_dialog import Ui_Dialog


class AboutUsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        img = QImage('material/icon.jpg')
        self.ui.label.setPixmap(QPixmap.fromImage(img))

