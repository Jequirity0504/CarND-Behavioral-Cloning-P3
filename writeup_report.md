# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image2]: ./writeup/center.jpg"
[image6]: ./writeup/image.jpg "Normal Image"
[image7]: ./writeup/flip_img.jpg "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The model I used finally is model published by NVIDIA Autonomous Car Group.

At very first, the data is normalized in the model using a Keras lambda layer (model.py line 47). 
Then I use a cropping layer(code line 49) to crop the images in Keras. 

Following them are five convolutional layers with 5x5 or 3x3 filter sizes and RELU activation function(model.py line 50 - 54).

After the five convolutional layers, there is a flatten layer to flatten parameter(model.py line 55)

And then , there are four fully connected layers with only one output at last(model.py line 56, 57, 60, 61)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 59). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 65). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 64).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I only used center images for network training. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find a well-known network as base and adapt it to make the vehicle driving as human does.

My first step was to use a convolution neural network model similar to the model published by autonomous vehicle team. I thought this model might be appropriate because the network NVIDIA autonomous team use it for training a real car to drive autonomously, it was a powerful network and it was used to solve similar problem.

Then I started the improve the model by preprocessing the data. I preprocessed the data by two steps: normalizing the data and mean centering the data. For normalization, I add a Lambda layer to my model, within the lambda layer, I normalize the image by dividing each element by 255, which is the maximum value of an image pixel, to normalize to a range between 0 and 1. Then I meaned center the image by subtracting 0.5 for each element to shift the element mean down to 0.

Another way to improve the model is to crop the images. As we can see, the top pixels of images mostly capture sky and trees, the bottom pixels consist the hood of the car, so I crop out these parts of images in Keras.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by add a dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. The car couldn't cross the corner after the bridge.

Then I use a bigger dropout rate , which make the model works well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 44-61) consisted of a convolution neural network with the following layers and layer sizes:

```sh
model.add(Cropping2D(cropping=((70,25), (0,0))))                              
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))	
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))	  
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
#Dropout layer
model.add(Dropout(0.5)) 
model.add(Dense(10))
model.add(Dense(1))
```

The network consists of a normalization layer, followed by five convolutional layers, followed by four fully connected layers.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. But after several round training, I found  the sample data is better because it could make the car always drive center. So I use the sample data Udacity provided.Here is an example of center lane driving:

![alt text][image2]


The final model architecture (model.py lines 44-61) consisted of a convolution neural network with the following layers.

To augment the datasat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]



After the collection process, I had 16072 number of data points. I then preprocessed this data by normalized and mean the image and cropping the images as talked before.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 because after 5 epochs the validation loss grow. I used an adam optimizer so that manually training the learning rate wasn't necessary.

