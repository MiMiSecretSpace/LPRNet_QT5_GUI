import cv2
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"  # exclude I, O
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}


class LPRNet(object):
    def __init__(self, model_filepath):
        # The file path of model
        self.model_filepath = model_filepath
        self.sess = None
        # Initialize the model
        self.load_graph(model_filepath=self.model_filepath)

    def load_graph(self, model_filepath):
        # print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            # Define input tensor
            self.input = tf.placeholder(
                tf.float32,
                shape=(None, 24, 94, 3),
                name='inputs')

            tf.import_graph_def(graph_def, {'inputs': self.input})

    def test(self, data):
        if not self.sess:
            self.sess = tf.InteractiveSession(graph=self.graph)
        image = cv2.resize(data, (94, 24))
        img_batch = np.expand_dims(image, axis=0)
        logits = self.graph.get_tensor_by_name('import/decoded:0')
        output = self.sess.run(logits, feed_dict={self.input: img_batch})
        return output

    def decode(self, numbers):
        for item in numbers:
            # print(item)
            expression = ['' if i == -1 else DECODE_DICT[i] for i in item]
            expression = ''.join(expression)
        return expression
