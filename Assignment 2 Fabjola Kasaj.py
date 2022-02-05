#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[2]:


from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
print(tf.__version__)


# # Import labels' csv file and convert it in a dataframe

# In[ ]:





# we will need this information as a dataframe to be able to read the categories. Based on these categories, later on, we are going to classify our images.

# In[ ]:





# In[3]:


import pandas as pd
df=pd.read_csv('C:/Users/STSC/Downloads/Orchid Flowers Dataset-v1.1/Orchid Flowers Dataset-v1.1/labels.csv')


# In[4]:


df.head()


# In the following code I have tried to structure our dataset. Since the original one, is just a folder with more than 7000 images in it but no subfolders, I have chosen this way to preprocess the data.
# I have grouped the dataframe by 'category' column. After that I have been working with both image's folder and label's folder. 
# 1- Create a new empty folder
# 2- Create new paths within this folder with the category ID.
# 3- Loop through original image folder. If the name of the image matches the image in the dataframe that image will join the new coresponding file with the corresponding category ID
# 
# 
# 

# In[5]:


ab=df.groupby('category')       


# In[5]:


print(ab.last())


# In[6]:


ab.get_group(10)


# In[7]:


#for j in range(1, 157):
#    for i in ab.get_group(j):
#        print(i)

print(ab.image.apply(list).reset_index(name='image'))

test = ab.image.apply(np.array).reset_index()
print("Test")
print(test.image)

for img in test.image:
    print(img)


# In[6]:


path = 'C:/Users/STSC/Downloads/test-kyle'
imgPath = 'C:/Users/STSC/Downloads/Orchid Flowers Dataset-v1.1/Orchid Flowers Dataset-v1.1/Orchid_Images'


# In[ ]:





# This part of code I am goind to mark it down. It is used to create the subfolders in my computer

# In[ ]:





# import os
# import shutil
# for i in test.category:
#     newPath = path + "/" + str(i)
#     os.mkdir(newPath)
#     
#     imagesArray = test.loc[test['category'] == i].image
#     for images in imagesArray:
#         for image in images:
#             imagePath = imgPath + "/" + str(image)
#             newImagePath = newPath + "/" + str(image)
#             shutil.copyfile(imagePath, newImagePath)
#             

# In[ ]:





# # Preprocessing

# In[ ]:





# Preprocessing is very important here. I have used ImageDataGenerator for augmentation process. It is going to rescale and generalize the data so the model can train better.
# I have used flow_from_directory() function since I already was able to organize the pictures into subfolders.

# In[ ]:





# In[7]:


data_generator = ImageDataGenerator(
    rescale = 1. / 255, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 180,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    validation_split = 0.2) 


# In[8]:


train_generator = data_generator.flow_from_directory(
    path, 
    target_size =(128, 128), 
    batch_size = 32,
    shuffle = True,
    class_mode = 'categorical',
    seed = 42,
    subset='training')


# In[9]:


validation_generator = data_generator.flow_from_directory( 
    path, 
    target_size =(128, 128), 
    batch_size = 32,
    shuffle = True,
    class_mode = 'categorical',
    seed = 42,
    subset='validation')


# In[ ]:





# # Split the data into train and validation

# In[ ]:





# In[25]:


from tqdm import tqdm


# In[26]:


# Store the data in X_train, y_train variables by iterating over the batches
batch_size=32
train_generator.reset()
X_train, y_train = next(train_generator)
for i in tqdm(range(int(len(train_generator)/batch_size)-1)): #1st batch is already fetched before the for loop.
  img, label = next(train_generator)
  X_train = np.append(X_train, img, axis=0 )
  y_train = np.append(y_train, label, axis=0)
print(X_train.shape, y_train.shape)


# In[27]:


# Store the data in X_train, y_train variables by iterating over the batches
batch_size=32
train_generator.reset()
X_valid, y_valid = next(train_generator)
for i in tqdm(range(int(len(train_generator)/batch_size)-1)): #1st batch is already fetched before the for loop.
  img, label = next(train_generator)
  X_valid = np.append(X_valid, img, axis=0 )
  y_valid = np.append(y_valid, label, axis=0)
print(X_valid.shape, y_valid.shape)


# In[ ]:





# # Data Visualization

# In[ ]:





# Cheking label's distribution ( 30 more commons one)

# In[16]:


import matplotlib.pyplot as plt
df.category.value_counts().iloc[ :30].plot(kind='bar')
plt.title('Labels counts')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()


# Plotting some of the images from training directory

# In[28]:


plt.figure(figsize=(12,12))
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i])
# show the figure
plt.show()


# In[ ]:





# # Build the model

# In[ ]:





# In[31]:


model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(156, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[32]:


model.summary()


# In[33]:


history = model.fit_generator(
        train_generator,
        epochs=20,
        validation_data=validation_generator)


# In[ ]:





# In[34]:


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(history.epoch, np.array(history.history['accuracy']),
           label='Train Accuracy')
  plt.plot(history.epoch, np.array(history.history['val_accuracy']),
           label = 'Val Accuracy')
  plt.legend()
  plt.ylim([0, 1])

plot_history(history)


# From the model and the graph itself we can see that the model is training. SInce the difference between train and validation is small this mean that the model is predicting correctly. 
# I am aware that the loss is high, and the accuracy is low to be able to call this a good model but it was taking very long to run the fitting function. Increasing the number of epochs and maybe adding hidden layers would have given a better result.

# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


history.history


# In[ ]:





# # Evaluate the validation set

# In[ ]:





# In[38]:


model.evaluate_generator(validation_generator, 128)


# In[ ]:


One more time, through the metrics, we can see that these features do noy indicated a good model


# In[ ]:





# # Predict an Image

# In[ ]:





# In[39]:


# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(X_valid[:3])
print("predictions shape:", predictions.shape)


# In[ ]:





# In[ ]:





# In[56]:


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[58]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], y_valid, X_valid)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# In[ ]:




