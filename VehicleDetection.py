from PyQt5.QtCore import pyqtSignal

from ObjectDetection import ObjectDetection

VEHICLE = ["person", "car", "motorcycle", "bus", "truck"]


class VehicleDetection(ObjectDetection):
    def __init__(self, model_filepath):
        super().__init__(model_filepath)

    def scoring(self, image):
        result = self.test(data=image, rearrange=False)
        img_height, img_width, img_channels = image.shape
        crop_images_pos = []
        for i in range(10):
            score = result[2][0][i]
            category = result[1][0][i]
            if category == 0 or category == 2 or category == 3 or category == 5 or category == 7:
                if score > 0.6:
                    top = abs(int(result[0][0][i][0] * img_height))
                    left = abs(int(result[0][0][i][1] * img_width))
                    bottom = abs(int(result[0][0][i][2] * img_height))
                    right = abs(int(result[0][0][i][3] * img_width))
                    crop_images_pos.append([left, top, right, bottom])#, VEHICLE[category]])

        return crop_images_pos
