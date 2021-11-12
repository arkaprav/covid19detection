#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import cv2


# In[3]:


img = image.load_img('C:\\xray_dataset_covid19\\train\\NORMAL\\IM-00001 (1).jpeg')


# In[4]:


plt.imshow(img)


# In[7]:


cv2.imread('C:\\xray_dataset_covid19\\train\\NORMAL\\IM-00001 (1).jpeg').shape


# In[8]:


train = ImageDataGenerator(rescale=1./255,
                           shear_range=0.2,
                           zoom_range=0.2,
                           horizontal_flip=True,
                           validation_split=0.2)
val = ImageDataGenerator(rescale = 1./255)


# In[9]:


train_dataset = train.flow_from_directory('C:\\xray_dataset_covid19\\train',
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode = 'binary')
val_dataset = val.flow_from_directory('C:\\xray_dataset_covid19\\val',
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode = 'binary')


# In[10]:


train_dataset.class_indices


# In[11]:


cnn = models.Sequential([
    layers.Conv2D(64,(3,3),activation='relu',input_shape=(200,200,3)),
    layers.MaxPool2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D(2,2),
    layers.GlobalMaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])


# In[12]:


from tensorflow.keras.optimizers import RMSprop
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[13]:


model_fit = cnn.fit(train_dataset,epochs=20,validation_data=val_dataset)


# In[17]:


import numpy as np
dir_path = 'C:\\xray_dataset_covid19\\New folder'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'\\'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    X=image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = cnn.predict(images)
    if val == 1:
        print(i+' Its Covid19')        
    else:
        print(i+' Its all right')

