import os
import csv
import cv2
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

lines = []
with open(os.path.join('..', 'merge_data', 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]  # data is created on Linux
    current_path = os.path.join('..', 'merge_data', 'IMG', filename)
    image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Cropping2D(cropping=((55,20), (0,0)), input_shape=(160,320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
#model.add(Conv2D(24,(5,5),activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(24,(5,5),activation='relu'))
#model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(36,(5,5),activation='relu'))
#model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(48,(5,5),activation='relu'))
#model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4, verbose=1)

model.save('modelNVidia.h5')

## plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.savefig(os.path.join('.', 'examples', 'train_data.png'))
