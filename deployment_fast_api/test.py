## Importing necessary Lib
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
plt.style.use('dark_background')
import warnings
warnings.filterwarnings("ignore")
import matplotlib.image as mpimg
np.random.seed(42)  # for reproducibility

import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
import keras

# import the library needed for one hot encoding
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import img_to_array, load_img

from tensorflow.keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.model_selection import train_test_split, KFold

global final_model
## Loading the model
final_model = load_model('mnist.h5')

final_model.summary()
## Evaluate Model
def test_results(X, y):

  score = final_model.evaluate(X, y, verbose=1)
  # Create a dictionary of model results
  print('Test_loss:', score[0])
  print('Test_accuracy:', score[1])

## Prediction
def predict_digit(path, ):

  img = load_img(path)
  #resize image to 28x28 pixels
  img = img.resize((28,28))
  #convert rgb to grayscale
  img = img.convert('L')
  img = np.array(img)
  #reshaping to support our model input and normalizing
  img = img.reshape(1,28,28,1).astype('float32')
  img = img/255.0
  #predicting the class
  res = final_model.predict([img])[0]
  return np.argmax(res), max(res)

# input_image_path = input('Path of the image to be predicted: ')
# a, b = predict_digit(input_image_path)
# print('Predicted Digit is: {}'.format(a))
# print('Maximum Probablity: {}'.format(b))

