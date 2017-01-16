# sdnd_p3_behaviour_cloning Read Me

##Background

This is repository is my implementation to solve project 3 of sdnd nanodegree on udacity. The goal of the project to to train a network to self drive a car using behaviour cloning methodology. The whole behaviour clonning experience here is provided by a simulator provided by udacity. The input data here are camera frames from the center camera of the simulated car being driven on track by a human, wchile the target outut is the steering angle being recorded for the corresponding frames. After training the network on the recorded data, the network is than put to test by allowing it to actually take control of the simulator and drive the car. 

However, as I lack a analog joystick to produce reasonable data for training, my source of the training data comes from udacity at
 - https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

##Methodology

### Data 

The data is first duplicated by flipping each center-camera vertically and negating the steering angle to generate a second set of data. Data is then pickled as data.p.

- The code to generate the flipped image is flip_jpg.py
- The code to generate the flipped data pooints to be appended to driving_log.csv is flip_csv.py
- The code to pickle the data is make_pickle.py

### Preprocessing

Input images are normalized linearly to be betwen -0.9 and 0.9.

### Validation/Testing/Training data split

First, 20% of the data is split out into testing data. Then out of the rest of the data remaining, 20% more is split into validation data. Leaving the rest to be training data.

Since sklearn train_test_split function is not meant to perform stratify split for regression training naturally. The input data is first split into 10 groups categorized by the magnitude of target steering angle before performing the split to ensure each of the the validation/testing/training is generated in a stratified fashion.

### Network Architecture

The architecture of the network is mostly based on the Nvidia paper "End to End Learning for Self-Driving Cars" 
 - https://arxiv.org/pdf/1604.07316v1.pdf

The Network consists of 9 layers, 
 - layer 1: Convolution Layer depth of 24 and max pooling factor of 2. Followed by batch normalization.
 - layer 2: Convolution Layer depth of 36 and max pooling factor of 2. Followed by batch normalization.
 - layer 3: Convolution Layer depth of 48 and max pooling factor of 2. Followed by batch normalization.
 - layer 4: Convolution Layer depth of 64 and max pooling factor of 2.
 - layer 5: Convolution Layer depth of 72 and max pooling factor of 2.
 - layer 6: Fully Connected Flat layer with 1164 nodes. With a dropout of 50%
 - layer 7: Fully Connected Flat layer with 100 nodes.
 - layer 8: Fully Connected Flat layer with 10 nodes.
 - layer 9: Fully Connected Output layer with 1 nodes.

### Training Parameter
 - training_epochs = 22
   - but best model (minumum val_loss) is selected and saved using keras.callbacks.ModelCheckpoint module which happened at epoc 21
 - batch_size = 128
 - dropout_rate = 0.50
 - learning_rate = 0.005

### Tuning and Some Observations

For some of the more difficult spots with sharper turns, data is duplicated a few more times (4 to 10 times) untill reasonable results are obtained.

1. elu activiation seems to help the network to train faster as compared to relu.  
2. he_normalization seems to result in a better performance.
3. Including some batch_normalizaion layers in the earlier layers also result in a better performance. Including batch_normalization in later layers actually seems to make things worse.
4. During my test runs, it seems that adding drop out layer to one of the more heavy layers (5 or 6) helps prevent over fitting. But adding to layer 7 or 8 seems to really slow down the training. 
5. While testing the model and observing the effects of tuning various parameters, smaller image size is used to speed up the process. 

####More details can be found on behaviour-cloning.ipynb, while the source code is also extracted out as behaviour-cloning.py
####A video can be found at: https://www.youtube.com/watch?v=VR7RftOcgQw&feature=youtu.be
