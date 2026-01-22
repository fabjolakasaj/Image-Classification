#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:



import keras 
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# # Generate images through ImageDataGenerator

# In[2]:


path = 'C:/Users/STSC/Downloads/test-kyle'

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   validation_split=0.2,
                                   horizontal_flip = True)
#test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(path,
                                                 target_size = (224, 224),
                                                 batch_size = 8,
                                                 subset="training",
                                                 class_mode = 'categorical')

validation_set = train_datagen.flow_from_directory(path,
                                                 target_size = (224, 224),
                                                 batch_size = 8,
                                                subset="validation",
                                                 class_mode = 'categorical')



#test_set = train_datagen.flow_from_directory(path,
                                            #target_size = (224, 224),
                                            #batch_size = 8,
                                            #class_mode = 'categorical')

#color_mode = "grayscale"


STEP_SIZE_TRAIN=training_set.n//training_set.batch_size
STEP_SIZE_VALID=validation_set.n//validation_set.batch_size
#STEP_SIZE_TEST=test_set.n//test_set.batch_size


# # Split the image data set into train and validation data

# In[3]:


from tqdm import tqdm


# In[4]:


# Store the data in X_train, y_train variables by iterating over the batches
batch_size=32
training_set.reset()
X_train, y_train = next(training_set)
for i in tqdm(range(int(len(training_set)/batch_size)-1)): #1st batch is already fetched before the for loop.
  img, label = next(training_set)
  X_train = np.append(X_train, img, axis=0 )
  y_train = np.append(y_train, label, axis=0)
print(X_train.shape, y_train.shape)


# In[5]:


# Store the data in X_train, y_train variables by iterating over the batches
batch_size=32
training_set.reset()
X_valid, y_valid = next(training_set)
for i in tqdm(range(int(len(training_set)/batch_size)-1)): #1st batch is already fetched before the for loop.
  img, label = next(training_set)
  X_valid = np.append(X_valid, img, axis=0 )
  y_valid = np.append(y_valid, label, axis=0)
print(X_valid.shape, y_valid.shape)


# # Load VGG16 model

# In[6]:


IMAGE_SIZE = [224, 224]
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#here [3] denotes for RGB images(3 channels)

#don't train existing weights
for layer in vgg.layers:
 layer.trainable = False
 
x = Flatten()(vgg.output)
prediction = Dense(156, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit_generator(training_set, 
                    steps_per_epoch=STEP_SIZE_TRAIN, 
                    epochs = 10, verbose=1, 
                    validation_data = validation_set, 
                    validation_steps = STEP_SIZE_VALID)


# In[ ]:





# # Vizualisation

# In[ ]:


plt.figure(figsize=(12,12))
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i])
# show the figure
plt.show()


# In[ ]:





# # Evaluating our model on validation data set

# In[ ]:


score = model.evaluate(validation_set)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy','Validation Accuracy','loss','Validation Loss'])
plt.show()


# As we can see, model's accuracy is really low and theres a big gap in betweeen loss functions. What I would try would be adding more hiden layers and increasing the number of epochs

# # Train the top layer

# In[ ]:





# In[ ]:


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss = keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.Accuracy()],
)

epochs = 10
model.fit_generator(training_set, epochs=epochs, validation_data=validation_set)


# # Do a round of fine-tuning of the entire model

# In[ ]:


# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss = keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.Accuracy()],
)

epochs = 10
model.fit_generator(training_set, epochs=epochs, validation_data=validation_set)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




