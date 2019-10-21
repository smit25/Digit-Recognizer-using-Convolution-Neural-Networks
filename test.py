# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
#from tensorflow.keras.utils import np_utils
from keras import utils
import cv2
"""
camera = cv2.VideoCapture(0)
for i in range(5):
	j=int(input("Enter"))
	if j==1:

        return_value, image = camera.read()
        cv2.imshow("test",image)
    if j==0:
    	break
del(camera)
camera.release()
cv2.destroyAllWindows()
"""
seed=7
np.random.seed(seed)
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train = x_train.reshape(60000,28,28,1).astype('float32')
x_test = x_test.reshape(10000,28,28,1).astype('float32')

x_train=x_train/255 # normalization
x_test=x_test/255
num_classes = 10
y_train = utils.to_categorical(y_train,num_classes)
y_test = utils.to_categorical(y_test,num_classes)
y_train[0]
	
model = Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), input_shape=( 28, 28,1), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32,activation ='relu'))
model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(num_classes, activation ='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	

model.fit(x_train,y_train, epochs=3,  verbose =2)
pred=model.predict(x_test)
loss,accuracy=model.evaluate(x_test,y_test, batch_size=32)
print('Test loss: %.4f accuracy: %.4f' % (loss, accuracy))
# FOR TESTING PURPOSE
first_20=np.argmax(pred, axis=1)[:20]
ans_20= np.argmax(y_test,axis =1)[:20]
print(first_20)
print(ans_20)
