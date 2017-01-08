
# coding: utf-8

# # Project 3: Behaviour Cloning

# ## Dataset
# 
# The dataset provided in project description is used. 
# Flipped images are also generated with the flipped entries in the .csv, so the data set is twice as big.

# ## First define some testing veriables

# In[1]:

test_on_small_data = False
#test_on_small_data = True

#resize_image = False
resize_image = True

print("start")


# ## Load the Data
# 
# Start by importing the data from the pickle file.

# In[2]:

import pickle
import os
import sys
import numpy as np
import scipy.misc as misc

#CUDA_VISIBLE_DEVICES=1

if test_on_small_data:
    data_folder = "small_data"
else:
    data_folder = "data"

print("Loading Image")
data_file = os.path.join(data_folder,"data.p")
with open(data_file, mode='rb') as f:
    all_data = pickle.load(f, encoding='latin1')
    #all_data = pickle.load(f)
    f.close()

split_test_validation_set = False

y_train = np.array(all_data["results"])
x_train = np.array(all_data["features"])

print("shape x_all:", x_train.shape)
print("shape y_all:", y_train.shape)


# In[3]:


def regression_train_test_split(x_train, y_train, 
                                random_state = False, test_size = 0.25, 
                                min_y = -1, max_y = 1, groups = 10 ):
    
    from sklearn.model_selection import train_test_split
    
    groupped_x = []
    groupped_y = []
    step_size = (max_y - min_y)/groups
    for g in range(groups):
        groupped_x.append([])
        groupped_y.append([])
    for i in range(len(y_train)):
        for g in range(groups):
            #min_y_bound = min_y + step_size*g
            max_y_bound = max_y - step_size*(groups-(g+1))
            if y_train[i] <= max_y_bound:
                groupped_x[g].append(x_train[i])
                groupped_y[g].append(y_train[i])
                break

    for g in range(groups):
        #print(g, np.array(groupped_x[g]).shape, np.array(groupped_y[g]).shape)
        if random_state:
            tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = train_test_split(
                                                                            np.array(groupped_x[g]),
                                                                            np.array(groupped_y[g]),
                                                                            test_size=test_size,
                                                                            random_state=random_state)
        else:
            tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = train_test_split(
                                                                            np.array(groupped_x[g]),
                                                                            np.array(groupped_y[g]),
                                                                            test_size=test_size)
        if g == 0:
            x_train = tmp_x_train
            x_test = tmp_x_test
            y_train = tmp_y_train
            y_test = tmp_y_test
        else:
            x_train = np.concatenate((x_train,tmp_x_train))
            x_test = np.concatenate((x_test,tmp_x_test))
            y_train = np.concatenate((y_train,tmp_y_train))
            y_test = np.concatenate((y_test,tmp_y_test))
    return x_train, x_test, y_train, y_test


# ## Split data into Train/Validation/Testing sets

# In[4]:

print("========= Before Splitting =========")
print("shape of x_train, ",x_train.shape)
print("shape of y_train, ",y_train.shape)

if not split_test_validation_set:
    if test_on_small_data:
        print("testing on small data set")
        x_test = x_train
        x_valid = x_train
        y_test = y_train
        y_valid = y_train
    else:
        x_train, x_test, y_train, y_test = regression_train_test_split(x_train,y_train,
                                                                    #random_state=10,
                                                                    test_size=0.2)
        x_train, x_valid, y_train, y_valid = regression_train_test_split(x_train,y_train, 
                                                                    #random_state=10,
                                                                    test_size=0.2)

print("========= After Splitting  =========")
print("shape of x_train, ",x_train.shape)
print("shape of y_train, ",y_train.shape)
print("shape of x_valid, ",x_valid.shape)
print("shape of y_valid, ",y_valid.shape)
print("shape of x_test, ",x_test.shape)
print("shape of y_test, ",y_test.shape)


# In[5]:

n_train = x_train.shape[0]
n_test = x_test.shape[0]
n_valid = x_valid.shape[0]

input_shape = x_train.shape[1:]


# In[6]:

####################################
# Training Parameters
training_epochs = 40
batch_size = 128
dropout_rate = 0.50
learning_rate = 0.005
####################################


# In[7]:

from alexnet import AlexNet
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Convolution2D
from keras.layers.core import Dense, Dropout, Activation, Reshape, Lambda, ActivityRegularization
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.engine.topology import Layer

keep_prob = tf.placeholder(tf.float32)


# In[8]:

'''
custom layer for resizing image to speed up training, 
Not used as saving of custom layer does not work (https://github.com/fchollet/keras/issues/2435)
example Usage:
    model.add( MyResizeImg(output_dim = (32,64,3)) )
'''
class MyResizeImg(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyResizeImg, self).__init__(**kwargs)
         
    def call(self, x, mask=None):
        return tf.image.resize_images(x, self.output_dim[0:2], method=0, align_corners=False)

    def get_output_shape_for(self, input_shape):
        output_shape = (input_shape[0], *self.output_dim)
        return output_shape


# In[9]:

model = Sequential()

if resize_image:
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (80,160)), input_shape=input_shape))
    model.add(Lambda(lambda x: x*1.8/255.0 - 1.))
else:
    model.add(Lambda(lambda x: x*1.8/255.0 - 1., input_shape=input_shape))
    
#hidden layer 1
model.add(Convolution2D(24, 5,5,
                        border_mode='same',
                        activation='elu',
                        init='he_normal'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())


#hidden layer 2
model.add(Convolution2D(36, 5,5, 
                        border_mode='same',
                        activation='elu',
                        init='he_normal'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())


#hidden layer 3
model.add(Convolution2D(48, 5,5, 
                        border_mode='same',
                        activation='elu',
                        init='he_normal'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())

#hidden layer 4
model.add(Convolution2D(64, 3,3, 
                        border_mode='same',
                        activation='elu',
                        init='he_normal'))
model.add(MaxPooling2D((2,2)))

#hidden layer 5
model.add(Convolution2D(64, 3,3, 
                        border_mode='same',
                        activation='elu',
                        init='he_normal'))
model.add(MaxPooling2D((2,2)))

#hidden layer 6
model.add(Flatten())
model.add(Dense(1164, activation='elu', init='he_normal'))
model.add((Dropout(dropout_rate)))

#hidden layer 7
model.add(Dense(100, activation='elu', init='he_normal'))

#hidden layer 8
model.add(Dense(50, activation='elu', init='he_normal'))

#hidden layer 9
model.add(Dense(10, activation='elu', init='he_normal'))

#output layer
model.add(Dense(1, activation='elu', init='he_normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

init = tf.global_variables_initializer()



# In[10]:

validation_accuracy = 0.0


print("training start")
# Launch the graph

history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=training_epochs,
                    verbose=1, validation_data=(x_valid, y_valid))

model.evaluate(x_test, y_test)


# **Test Accuracy:** (fill in here)

# ## Save Model

# In[12]:

## save model
from keras.models import load_model
import json

with open('model.json', 'w') as outfile:
    json_model = model.to_json()
    json.dump(json_model, outfile)
    outfile.close()

model.save('model.h5')


# ## Design Notes and Obervatinos
# 
# 
# The source of the training data comes from udacity at (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
# 
# ### Data 
# 
# The data is first duplicated by flipping each center-camera vertically and negating the steering angle to generate a second set of data. Data is then pickled as data.p.
# 
# - The code to generate the flipped image is flip_jpg.py
# - The code to generate the flipped data pooints to be appended to driving_log.csv is flip_csv.py
# - The code to pickle the data is make_pickle.py
# 
# ### Preprocessing
# Input images are normalized linearly to be betwen -0.9 and 0.9.
# 
# 
# ### Network Architecture
# 
# The architecture of the network is mostly based on the Nvidia paper "End to End Learning for Self-Driving Cars" (https://arxiv.org/pdf/1604.07316v1.pdf)
# 
# The Network consists of 9 layers, 
#  - layer 1: Convolution Layer depth of 24 and max pooling factor of 2. Followed by batch normalization.
#  - layer 2: Convolution Layer depth of 36 and max pooling factor of 2. Followed by batch normalization.
#  - layer 3: Convolution Layer depth of 48 and max pooling factor of 2. Followed by batch normalization.
#  - layer 4: Convolution Layer depth of 64 and max pooling factor of 2.
#  - layer 5: Convolution Layer depth of 64 and max pooling factor of 2.
#  - layer 6: Fully Connected Flat layer with 1164 nodes. With a dropout of 50%
#  - layer 7: Fully Connected Flat layer with 100 nodes.
#  - layer 8: Fully Connected Flat layer with 10 nodes.
#  - layer 9: Fully Connected Output layer with 1 nodes.
# 
# ### Tuning and Some Observations
# 
# For some of the more difficult spots with sharper turns, data is duplicated a few more times (4 to 10 times) untill reasonable results are obtained.
# 
# 1. elu activiation seems to help the network to train faster as compared to relu.  
# 2. he_normalization seems to result in a better performance.
# 3. Including some batch_normalizaion layers in the earlier layers also result in a better performance. Including batch_normalization in later layers actually seems to make things worse.
# 4. During my test runs, it seems that adding drop out layer to one of the more heavy layers (5 or 6) helps prevent over fitting. But adding to layer 7 or 8 seems to really slow down the training. 
# 5. While testing the model and observing the effects of tuning various parameters, smaller image size is used to speed up the process. 
# 

# In[ ]:



