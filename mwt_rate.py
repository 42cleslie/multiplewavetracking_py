import random
from keras.engine.topology import Layer
import keras.backend as K
from skimage import transform

import numpy as np

class SppnetLayer(Layer):
    '''This layer takes an input tensor and pools the tensor
      in local spatial bins.
      This layer uses Max pooling.
      It accepts input in tensorflow format. # channels last

    # Input
        list of filter in form [x,y,z] 
    # Input shape : 4d tensor [None, X,Y, channels]
    # Output shape : 3d tensor [None,pooled dim, channels] 

    '''
    def __init__(self, filters = [1], **kwargs):
        self.filters = filters
        super(SppnetLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        length = 0
        for f_size in self.filters:
            length+= (f_size*f_size)
        return (input_shape[0],length*input_shape[3])
      
    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(SppnetLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
      output = []
      for f_size in self.filters:
        win_size = K.int_shape(inputs)[1]/f_size
        #print(win_size)
        win_size = int(win_size)
        for x_start in range(0,f_size):
          for y_start in range(0,f_size):
            X = int(x_start*win_size)
            Y = int(y_start*win_size)
            result = K.max(inputs[:,X:X+win_size,Y:Y+win_size,:],axis = (1,2))
            output.append(result)
      output = K.concatenate(output)
      return output

def rate(ratings, wave_list, model):
    for wave in wave_list:
        for frame in wave.frame_data:
            # resize the image so it is the same size as the training data
            np_image = np.array(frame).astype('float32')/255
            np_image = transform.resize(np_image, (224, 224, 3))
            np_image = np.expand_dims(np_image, axis=0)
            ratings.append(np.argmax(model.predict(np_image)))
        wave.frame_data = []


def placeholder(wave_frame):
    return random.randint(1, 10)


def get_final_rating(ratings):
    top_five_per = []
    min = 0

    if len(ratings) == 0:
        print ("Error: no ratings to analyze")
        return 1

    if len(ratings) < 5:
        return max(ratings)

    tenth_percentile = len(ratings) / 5

    for r in ratings:
        if len(top_five_per) < tenth_percentile:
            top_five_per.append(r)
        else:
            for ttr in top_five_per:
                if r > ttr:
                    top_five_per.remove(ttr)
                    top_five_per.append(r)
                    break
    print(top_five_per)

    return sum(top_five_per) / len(top_five_per)
