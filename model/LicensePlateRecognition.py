from model.ObjectDetection import ObjectDetection


class LicensePlateRecognition(ObjectDetection):
    def __init__(self, model_filepath):
        super().__init__(model_filepath)

    def scoring(self, image):
        result = self.test(data=image, rearrange=True)
        img_height, img_width, img_channels = image.shape
        crop_images_pos = []
        for i in range(1):
            #if score > 0.001:
            top = abs(int(result[0][0][i][0] * img_height))
            left = abs(int(result[0][0][i][1] * img_width))
            bottom = abs(int(result[0][0][i][2] * img_height))
            right = abs(int(result[0][0][i][3] * img_width))
            crop_images_pos.append([left, top, right, bottom])

        return crop_images_pos
