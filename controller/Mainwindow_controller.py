import cv2
import numpy as np

from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QMainWindow, QAction, QActionGroup, QDialog
from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtMultimedia import QCameraInfo

from view.Mainwindow_ui import Ui_MainWindow
from view.About_us_dialog import Ui_Dialog
from model.LPRNet import LPRNet
from model.VehicleDetection import VehicleDetection
from model.LicensePlateRecognition import LicensePlateRecognition
from controller.Utils import Utils
from controller.Detection import Detection
from controller.About_us_controller import AboutUsDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.image = np.ndarray
        self.lprnet_model = LPRNet('material/LPRNet.pb')
        self.vehicle_detection_model = VehicleDetection('material/ObjectModel.tflite')
        self.lp_recognition_model = LicensePlateRecognition('material/LicensePlateRecognition.tflite')
        self.label = None
        self.cap = None
        self.fps = 0
        self.timer = QTimer(self)
        self.thread = QThread()
        self.plates = None
        self.last_time = 0
        self.camera_group = QActionGroup(self)
        self.init_slots()
        self.show_availableCamera()

    def init_slots(self):
        self.ui.actionLPRNet.triggered.connect(self.set_lprnet_model)
        self.ui.actionObject_Detection.triggered.connect(self.set_vehicle_detection_model)
        self.ui.actionLicense_Plate_Recognition.triggered.connect(self.set_license_plate_recognition_model)
        self.ui.actionAbout_Us.triggered.connect(self.about_us)
        self.ui.Image_path.clicked.connect(self.set_image)
        self.ui.Video_path.clicked.connect(self.set_video)
        self.ui.play_buttom.clicked.connect(self.play_pause)
        self.ui.checkBox.stateChanged.connect(self.set_camera)
        self.camera_group.triggered.connect(lambda: self.ui.checkBox.setEnabled(True))
        #self.ui.menuCamera.aboutToShow.connect(self.show_availableCamera)
        self.timer.timeout.connect(self.timer_tick)

    def about_us(self):
        d = AboutUsDialog()
        d.exec_()

    def closeEvent(self, a0: QCloseEvent) -> None:
        if self.cap is not None:
            self.cap.release()
            self.thread.deleteLater()
            print('release')
        print('close')

    def show_availableCamera(self):
        self.ui.menuCamera.clear()
        QCameraInfo.defaultCamera().deviceName()
        for index, c in enumerate(QCameraInfo.availableCameras()):
            action = QAction(self)
            action.setCheckable(True)
            action.setText(c.description())
            action.setWhatsThis(str(index))
            self.camera_group.addAction(action)
            self.ui.menuCamera.addAction(action)

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
        for plate in self.plates:
            if plate[1] is '':
                break
            Utils.bounding_box(self, self.image, plate[0][0],
                               plate[0][1], plate[0][2],
                               plate[0][3], plate[1])

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

    def set_image(self):
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

    def set_video(self):
        ofd = Utils.open_file_dialog(self, '*.mp4, *.avi(*.mp4 *.avi)')
        if not ofd:
            return
        self.cap = cv2.VideoCapture(ofd[0])
        self.ui.video_path_text.setText(ofd[0])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def set_camera(self):
        camera = self.camera_group.checkedAction()
        if camera is None:
            print('Please select a camera')
            return
        if self.ui.checkBox.isChecked():
            self.cap = cv2.VideoCapture(int(camera.whatsThis()))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.ui.video_path_text.setText(camera.text())
        else:
            self.cap.release()
