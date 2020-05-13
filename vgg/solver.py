from mlpm.solver import Solver
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

class ImageClassificationSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        # Do you Init Work here
        self.model = VGG16(weights="./pretrained/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

        self.ready()
    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        # load an image from file
        image = load_img(data['input_file_path'], target_size=(224, 224))
        
        # convert the image pixels to a numpy array
        image = img_to_array(image)

        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # prepare the image for the VGG model
        image = preprocess_input(image)

        # predict the probability across all output classes
        yhat = self.model.predict(image)

        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]
        # print the classification
        print('%s (%.2f%%)' % (label[1], label[2]*100))

        result = {
            "label": label[1],
            "confidence": label[2]*100
        }

        return result # return a dict