import tensorflow as tf
import cv2
import numpy as np

from PyQt5.QtCore import QObject


class ObjectDetection(QObject):

    def __init__(self, model_filepath):
        self.model_filepath = model_filepath
        self.set_model()

    def set_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_filepath)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def test(self, data, rearrange):
        height = self.input_details[0]['shape'][1]
        width = self.input_details[0]['shape'][2]
        resize_img = cv2.resize(data, (width, height))
        input_data = np.expand_dims(resize_img, axis=0)
        if rearrange is True:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = [self.interpreter.get_tensor(self.output_details[0]['index']),
                       self.interpreter.get_tensor(self.output_details[1]['index']),
                       self.interpreter.get_tensor(self.output_details[2]['index']),
                       self.interpreter.get_tensor(self.output_details[3]['index'])]
        return output_data
