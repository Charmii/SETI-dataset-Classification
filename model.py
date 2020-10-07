import Tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
np.random.seed(42)
import warnings;warnings.simplefilter('ignore')
%matplotlib inline

train_images = pd.read_csv('images.csv')
train_labels = pd.read_csv('labels.csv')
val_images = pd.read_csv('val_images.csv')
val_labels = pd.read_csv('val_labels.csv')

train_images.shape #(3199, 8192)
val_images.shape #(799, 8192)

xtrain = train_images.values.reshape(3199,64,128,1)
xval = val_images.values.reshape(799,64,128,1)
ytrain = train_labels.values
yval = val_labels.values

plt.figure(0, figsize = (12,12))
for i in range(1,4):
    plt.subplot(1,3,i)
    num = np.random.randint(0, xtrain.shape[0])
    img = np.squeeze(xtrain[num])
    plt.imshow(img)
    plt.xlabel(str(num))


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
data_train = ImageDataGenerator(horizontal_flip = True)
data_val = ImageDataGenerator(horizontal_flip = True)

import keras
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(5,5),input_shape = (64,128,1)))
model.add(keras.layers.MaxPooling2D((3,3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(64,(5,5)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(4,activation = 'softmax'))

