import os
import csv
import cv2
import numpy as np
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout, concatenate, AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

def inception_module(x, params, data_format, concat_axis,
                     activation='relu',
                     padding='same'):

    # https://gist.github.com/nervanazoo/2e5be01095e935e90dd8  #
    # file-googlenet_neon-py

    (branch1, branch2, branch3, branch4) = params

    pathway1 = Conv2D(branch1[0], (1, 1),
                             activation=activation,
                             padding=padding,
                             use_bias=False,
                             data_format=data_format)(x)

    pathway2 = Conv2D(branch2[0], (1, 1),
                             activation=activation,
                             padding=padding,
                             use_bias=False,
                             data_format=data_format)(x)
    pathway2 = Conv2D(branch2[1], (3, 3),
                             activation=activation,
                             padding=padding,
                             use_bias=False,
                             data_format=data_format)(pathway2)

    pathway3 = Conv2D(branch3[0], (1, 1),
                             activation=activation,
                             padding=padding,
                             use_bias=False,
                             data_format=data_format)(x)
    pathway3 = Conv2D(branch3[1], (5, 5),
                             activation=activation,
                             padding=padding,
                             use_bias=False,
                             data_format=data_format)(pathway3)

    pathway4 = MaxPooling2D(pool_size=(1, 1), data_format=data_format)(x)
    pathway4 = Conv2D(branch4[0], (1, 1),
                             activation=activation,
                             padding=padding,
                             use_bias=False,
                             data_format=data_format)(pathway4)

    return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)

def create_googlenet():
    input = Input(shape=(160, 320, 3))
    crop = Cropping2D(cropping=((50,20), (0,0)))(input)
    x = Lambda(lambda x: x / 255.0 - 0.5)(crop)
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
    x = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
    # inception 3a
    x = inception_module(x, params=[(64, ), (96, 128), (16, 32), (32, )],
                     data_format='channels_last', concat_axis=-1)
    # inception 3b
    x = inception_module(x, params=[(128, ), (128, 192), (32, 96), (64, )],
                     data_format='channels_last', concat_axis=-1)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
    
    # inception 4a
    x = inception_module(x, params=[(192, ), (96, 208), (16, 48), (64, )],
                     data_format='channels_last', concat_axis=-1)
    # inception 4b
    x = inception_module(x, params=[(160, ), (112, 224), (24, 64), (64, )],
                     data_format='channels_last', concat_axis=-1)
    # inception 4c
    x = inception_module(x, params=[(128, ), (128, 256), (24, 64), (64, )],
                     data_format='channels_last', concat_axis=-1)
    # inception 4d
    x = inception_module(x, params=[(112, ), (144, 288), (32, 64), (64, )],
                     data_format='channels_last', concat_axis=-1)
    # inception 43
    x = inception_module(x, params=[(256, ), (160, 320), (32, 128), (128, )],
                     data_format='channels_last', concat_axis=-1)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')(x)
    
    # inception 5a
    x = inception_module(x, params=[(256, ), (160, 320), (32, 128), (128, )],
                     data_format='channels_last', concat_axis=-1)
    # inception 5b
    x = inception_module(x, params=[(384, ), (192, 384), (48, 128), (129, )],
                     data_format='channels_last', concat_axis=-1)
    
    x = AveragePooling2D(strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='linear')(x)
    
    model = Model(input=input, output=x)

    return model
    
# MAGIC
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
##

lines = []
with open(os.path.join('..', 'data', 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]  # data is created on Linux
    current_path = os.path.join('..', 'data', 'IMG', filename)
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = create_googlenet()
model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)

model.save('modelGoogleNet.h5')

