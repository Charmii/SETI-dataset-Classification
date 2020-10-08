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



initial_lr = 0.005
lrschedule = keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate = initial_lr,
decay_steps = 5,
decay_rate = 0.96,
staircase = True,
)
optimizer = keras.optimizers.Adam(learning_rate = lrschedule)


model.compile(optimizer = optimizer,
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])



batch_size = 32
history = model.fit_generator(datagen_train.flow(xtrain,ytrain,batch_size = batch_size,shuffle = True),
                    steps_per_epoch = len(xtrain)//batch_size,
                    validation_data = datagen_val.flow(xval,yval,batch_size=batch_size, shuffle = True),
                    validation_steps = len(xval)//batch_size,
                    epochs = 15,
                   )


model.evaluate(xval,yval)
#25/25 [==============================] - 2s 85ms/step - loss: 0.5731 - accuracy: 0.6821
#[0.5730670094490051, 0.682102620601654]

y_true = np.argmax(yval,1)
y_pred = np.argmax(model.predict(xval),1)
print(sklearn.metrics.classification_report(y_true,y_pred))
#              precision    recall  f1-score   support
#           0       0.81      0.99      0.89       199
#           1       0.50      0.28      0.36       200
#           2       0.52      0.45      0.48       200
#           3       0.75      1.00      0.85       200
#    accuracy                           0.68       799
#   macro avg       0.64      0.68      0.65       799
#weighted avg       0.64      0.68      0.65       799

print(sklearn.metrics.accuracy_score(y_true,y_pred)) #0.6821026282853567


