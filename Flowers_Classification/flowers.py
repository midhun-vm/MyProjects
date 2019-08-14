#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[53]:


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


# In[ ]:





# In[6]:


# Set the path of the input folder 

data = "E:/Projects/Flowers"

# List out the directories inside the main input folder

folders = os.listdir(data)

print(folders)


# In[27]:


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


# In[29]:


img = cv2.imread('E:/Projects/Flowers/daisy/5547758_eea9edfd54_n.jpg')


# In[30]:


img.shape


# In[22]:


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[51]:


label_dummies = pandas.get_dummies(train_labels)

labels =  label_dummies.values.argmax(1)
#pandas.unique(train_labels)
pandas.unique(labels)


# In[32]:


union_list = list(zip(train, labels))
random.shuffle(union_list)


# In[33]:


train,labels = zip(*union_list)


# In[34]:


# Convert the shuffled list to numpy array type

train = np.array(train)
labels = np.array(labels)


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(train,labels,test_size=0.25,random_state=42)


# In[36]:


train[0].shape


# In[46]:


cv2.imshow('image',x_test[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[52]:


y_test[0]


# In[65]:


# Develop a sequential model using tensorflow keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128,128,3)),
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])


# In[66]:


# Compute the model parameters

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[67]:


batch_size=128
epochs=50

#History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
 #                             epochs = epochs, validation_data = (x_test,y_test),
  #                            verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
History=model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))

