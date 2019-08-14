
# Import the necessary libraries

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random
from sklearn.model_selection import train_test_split

#dl libraraies
import numpy as np
from os import listdir
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import random as rn


# Set the path of the input folder 

data = 

# List out the directories inside the main input folder

folders = os.listdir(data)

print(folders)
# Import the images and resize them to a 128*128 size
# Also generate the corresponding labels

image_names = []
train_labels = []
train_images = []

size = 128,128

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue
      


# In[28]:


# Transform the image array to a numpy type

train = np.array(train_images)

train.shape

img = cv2.imread('')


img.shape


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


label_dummies = pandas.get_dummies(train_labels)

labels =  label_dummies.values.argmax(1)
#pandas.unique(train_labels)
pandas.unique(labels)
union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

# Convert the shuffled list to numpy array type

train = np.array(train)
labels = np.array(labels)
x_train,x_test,y_train,y_test=train_test_split(train,labels,test_size=0.25,random_state=42)

train[0].shape


# In[46]:


cv2.imshow('image',x_test[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

y_test[0]


# Develop a sequential model using tensorflow keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128,128,3)),
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

# Compute the model parameters

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




batch_size=128
epochs=50

#History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
 #                             epochs = epochs, validation_data = (x_test,y_test),
  #                            verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
History=model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))

