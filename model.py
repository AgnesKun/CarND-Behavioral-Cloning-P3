import os
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

# show summary of data
def show_hist(samples, title, file):
    y = []
    for line in samples:
        y.append(float(line[3]))
    plt.xlim(-1.0, 1.0)
    plt.hist(y)
    plt.xlabel('angle')
    plt.title(title)
    plt.savefig(file)
    plt.cla()

# generate center/left/right image and flipped center image
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # read image from center camera
                name = os.path.join('..', 'merge_data', 'IMG', batch_sample[0].split('/')[-1])
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # flip image from center camera
                image_flipped = np.fliplr(center_image)
                angle_flipped = -center_angle
                images.append(image_flipped)
                angles.append(angle_flipped)
                # read image from left camera
                lname = os.path.join('..', 'merge_data', 'IMG', batch_sample[1].split('/')[-1])
                left_image = cv2.cvtColor(cv2.imread(lname), cv2.COLOR_BGR2RGB)
                left_angle = center_angle + 0.2
                images.append(left_image)
                angles.append(left_angle)
                # read image from right camera
                rname = os.path.join('..', 'merge_data', 'IMG', batch_sample[2].split('/')[-1])
                right_image = cv2.cvtColor(cv2.imread(rname), cv2.COLOR_BGR2RGB)
                right_angle = center_angle - 0.2
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def get_samples():
    samples = []
    with open(os.path.join('..', 'merge_data', 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, shuffle=True, test_size=0.2)
    return train_samples, validation_samples

def create_model():
	model = Sequential()
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3)))
	model.add(Lambda(lambda x: x / 255.0 - 0.5))
	model.add(Conv2D(24,(5,5),activation='relu'))
	model.add(Dropout(0.1))
	model.add(MaxPooling2D())
	model.add(Conv2D(36,(5,5),activation='relu'))
	model.add(Dropout(0.1))
	model.add(MaxPooling2D())
	model.add(Conv2D(48,(5,5),activation='relu'))
	model.add(Dropout(0.1))
	model.add(MaxPooling2D())
	model.add(Conv2D(64,(3,3),activation='relu'))
	model.add(Dropout(0.1))
	model.add(Conv2D(64,(3,3),activation='relu'))
	model.add(Dropout(0.1))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	
	model.compile(loss='mse', optimizer='adam')

	return model

# get train, valid data
train_samples, validation_samples = get_samples()

# show histgram
show_hist(train_samples, 'Train Data', os.path.join('.', 'examples', 'train.png'))
show_hist(validation_samples, 'Validation Data', os.path.join('.', 'examples', 'valid.png'))

nbatch = 8
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=nbatch)
validation_generator = generator(validation_samples, batch_size=nbatch)

# create model
model = create_model()
model.summary()

# train data
earlystopper  = EarlyStopping(patience=4, verbose=1)
checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/nbatch,
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)/nbatch, epochs=16,
                    callbacks=[earlystopper, checkpointer])

# save trained model
#model.save('model.h5')

## plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.savefig(os.path.join('.', 'examples', 'train_data.png'))

