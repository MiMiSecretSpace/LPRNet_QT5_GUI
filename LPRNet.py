import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class LPRNet(object):
    def __init__(self, model_filepath):
        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath=self.model_filepath)

    def load_graph(self, model_filepath):
        #print('Loading model...')
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

        #print('Model loading complete!')
        self.sess = tf.InteractiveSession(graph=self.graph)

    def test(self, data):
        logits = self.graph.get_tensor_by_name('import/decoded:0')
        output = self.sess.run(logits, feed_dict={self.input: data})
        return output
