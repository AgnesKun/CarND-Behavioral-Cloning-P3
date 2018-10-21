# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

**Environment**

* Intel i7 2.4GHz, 8GB RAM, Windows10, GTX 660M   
* Tensorflow: 1.11 Keras: 2.24


[//]: # (Image References)

[image1]: ./examples/train_data.png "Train Data"
[image2.1]: ./examples/center_2016_12_01_13_30_48_287.jpg "Orginal Center Image"
[image2.2]: ./examples/left_2016_12_01_13_30_48_287.jpg "Orginal Left Image"
[image2.3]: ./examples/right_2016_12_01_13_30_48_287.jpg "Orginal Right Image"
[image3.1]: ./examples/mycenter_2016_12_01_13_30_48_287.jpg "Flipped Center Image"
[image3.2]: ./examples/myleft_2016_12_01_13_30_48_287.jpg "Flipped Left Image"
[image3.3]: ./examples/myright_2016_12_01_13_30_48_287.jpg "Flipped Right Image"
[image4.1]: ./examples/center_2018_10_19_21_31_36_589.jpg "Opposite Direction Center Image"
[image4.2]: ./examples/center_2018_10_19_21_31_36_589.jpg "Opposite Direction Left Image"
[image4.3]: ./examples/center_2018_10_19_21_31_36_589.jpg "Opposite Direction Right Image"
[image5.1]: ./examples/mycenter_2018_10_19_21_31_36_589.jpg "Flipped Opposite Direction Center Image"
[image5.2]: ./examples/mycenter_2018_10_19_21_31_36_589.jpg "Flipped Opposite Direction Left Image"
[image5.3]: ./examples/mycenter_2018_10_19_21_31_36_589.jpg "Flipped Opposite Direction Right Image"
[image6]: ./examples/train.png "Train Data Sets"
[image7]: ./examples/valid.png "Validation Data SEts"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

This is a summary statistics of original data set:

* The size of original data set is 8036

Also this is a summary statistics of data set of my driving in opposite direction:
* The size of my driving data set is 3643  

And model.py added a flipped image, a left image with angle + 0.2, a right image with angle - 0 in each images on the fly.
The total data set is:
* The total size of original data set with augmented image is 32144 
* The total size of my driving data set with augmented image is 14572  


#### 2.Exploratory visualization of the dataset.

I split the dataset into Train 80% and Valid 20%.  Here is an exploratory visualization of the data set before augmentation. This is a histogram showing how the train data is.  The data set has more negative angle because original data set has counter-clockwise direction.

![alt text][image6]

This is a validation data set before augmentation.

![alt text][image7]


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on nVidea model with added Dropout. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with 0.5 every after Full Connecting Network and with 0.1 every after CNN in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (create_model()).
And I used Checkpoint with saving best only. This is a learning curb.
![alt text][image1]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (create_model in model.py line 141).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First I tested LeNet, nVidia, AlexNet, VGG, and GoogLeNet, first (cloneleNet.py, cloneNVidia.py, cloneAlexNet.py, cloneVGG.py, and cloneGoogleNet.py).  LeNet and nVidia were OK, but AlexNet and VGG have a big network and I could not train them on my PC.  I could train them on Linux, but H5 files look inconmatible to my pc even if I have the same version of Keras and Tensorflow.  Also I could not train GoogleNet because of issue of TensorFlow on MKL-DNN.  
At this first stage, nVidia was the best among those 5 networks, because of its size and accuracy.

So I decided to use nVidia model with some tuning.  I added regularization, but it made worse, but Dropout made a little better.  I added Dropout with 0.1 after CNN, and Dropout with 0.5 after Fully Connected Network.
But car didn't turn on sharp turn with any tuning.  So I decided to add more images.

I added flipped image, and images from left and right Camera with 0.2 adjustment.  And I changed to use generator because of many data.  Then it finally could run on whole course.  But it didn't turn at a sharp turn with speed 25.

So I added images of driving with opposite direction.  But it didn't improve a lot.  I wanted to add data of track 2.  But it was not to easy to drive on track 2.  And I also wanted to add images when I drive at hight speed.  But it is also not eash to make it.  So I stopped here.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (create_model in model.py):

|Layer (type)  |               Output Shape |             Param #|
|--------------|----------------------------|--------------------|
|cropping2d_1 (Cropping2D)    |(None, 90, 320, 3)        |0      |
|lambda_1 (Lambda)            |(None, 90, 320, 3)        |0      |
|conv2d_1 (Conv2D)            |(None, 86, 316, 24)       |1824   |
|dropout_1 (Dropout)          |(None, 86, 316, 24)       |0      |
|max_pooling2d_1 (MaxPooling2 |(None, 43, 158, 24)       |0      |
|conv2d_2 (Conv2D)            |(None, 39, 154, 36)       |21636  |
|dropout_2 (Dropout)          |(None, 39, 154, 36)       |0      |
|max_pooling2d_2 (MaxPooling2 |(None, 19, 77, 36)        |0      |
|conv2d_3 (Conv2D)            |(None, 15, 73, 48)        |43248  |
|dropout_3 (Dropout)          |(None, 15, 73, 48)        |0      |
|max_pooling2d_3 (MaxPooling2 |(None, 7, 36, 48)         |0      |
|conv2d_4 (Conv2D)            |(None, 5, 34, 64)         |27712  |
|dropout_4 (Dropout)          |(None, 5, 34, 64)         |0      |
|conv2d_5 (Conv2D)            |(None, 3, 32, 64)         |36928  |
|dropout_5 (Dropout)          |(None, 3, 32, 64)         |0      |
|flatten_1 (Flatten)          |(None, 6144)              |0      |
|dense_1 (Dense)              |(None, 100)               |614500 |
|dropout_6 (Dropout)          |(None, 100)               |0      |
|dense_2 (Dense)              |(None, 50)                |5050   |
|dropout_7 (Dropout)          |(None, 50)                |0      |
|dense_3 (Dense)              |(None, 10)                |510    |
|dropout_8 (Dropout)          |(None, 10)                |0      |
|dense_4 (Dense)              |(None, 1)                 |11     |
|--------------|----------------------------|--------------------|
Total params: 751,419
Trainable params: 751,419

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used sample data, and left image with angle - 0.2, and right image with angle + 0.2 :

![alt text][image2.1]![alt text][image2.2]![alt text][image2.3]


Then I added a flipped image of center, left with angle - 0.2, and right with angle + 0.2:

![alt text][image3.1]![alt text][image3.2]![alt text][image3.3]

Then I added the driving with opposite direction:
![alt text][image4.1]![alt text][image4.2]![alt text][image4.3]


Then I added a flipped image of the driving with opposite direction of center, left with angle - 0.2, and right with angle + 0.2:
![alt text][image5.1]![alt text][image5.2]![alt text][image5.3]


I randomly shuffled the data set and put 20% of the data into a validation set. 

Then I made a generator to convey 46712 number of data to model by fip, left and right. I then preprocessed this data.  I cropped images 50 rows from top, and 20 rows from bottom.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 16, and I used Checkpoint with saving best only. I used an adam optimizer so that manually training the learning rate wasn't necessary.