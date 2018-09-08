# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/mymodel.png "Model Visualization"
[image2]: ./pictures/normal_driving.gif "Normal Driving"
[image3]: ./pictures/avoiding_crash_left.gif "Recovery Image"
[image4]: ./pictures/avoiding_crash_right.gif "Recovery Image"
[image5]: ./pictures/original_captured_image.jpg "Normal Image"
[image6]: ./pictures/flipped_captured_image.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to define model
* model_execution.py containing the script to read data, execute data augmentation, train my model with validation data.
* generator.py including the script for generating image data according to batch size given as argument
* data_augment.py including function for executing data augmentation with flip function
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* autonomous_driving.mp4 for demonstrating my model to drive on the road without any trouble

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for defining function which builds my model  including the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The `model_execution.py` file includes the code which executes the following:  
* load `driving_log.csv` file,  
* call `data_augment()` function which adds flipped images for training,  
* set generator as `train_generator` and `validation_generator`,  
* load my model with `build_mymodel()` function,  
* visualize and save my model to `mymodel.png`
* and save my model after comipiling and training.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of several convolution neural network and  fully-connected network mainly. I built my model which is inspired by VGG. My model resembles the structure of VGG and have completely same approach of VGG which has very deep netrowk layers. 
Convoluation neural network layers are constructed as follows  

1. 2 sequence of convolution network layers 3x3 filter size and depth 6, Batch normalization, and LeRU activation (code line 23-26 in `model.py`)
2. Max pooling whose pool size is 2x2 (code line 28 in `model.py`)
3. 2 sequence of convolution network layers who have 3x3 filter size and depth 6, Batch normalization, and LeRU activation (code line 31-34 in `model.py`)
4. Max pooling whose pool size is 2x2 (code line 36 in `model.py`)
5. 3 sequence of convolution network layers 3x3 filter size and depth 24, Batch normalization, and LeRU activation (code line 39-42 in `model.py`)
6. Max pooling whose pool size is 2x2 (code line 44 in `model.py`)
7. A convolution network layer 3x3 filter size and depth 96, Batch normalization, and LeRU activation (code line 47-49 in `model.py`)
8. Flatten function to align input data for fully-connetcted networks (code line 52 in `model.py`)
9. 2 sequence of fully-connected network layers whose output size is 800, Batch normalization, LeRU activation, and dropout whose keep probability is 0.5 (code line 55-64 in `model.py`)
10. A fully-connected network layers whose output size is 600, Batch normalization, LeRU activation, and dropout whose keep probability is 0.5 (code line 67-70 in `model.py`)
11. A fully-connected network layers whose output size is 300, Batch normalization, LeRU activation, and dropout whose keep probability is 0.5 (code line 73-76 in `model.py`)
12. A fully-connected network layers whose output size is 120, Batch normalization, LeRU activation, and dropout whose keep probability is 0.5 (code line 79-82 in `model.py`)
13. A fully-connected network layers whose output size is 1, and this output is regarded as predicted steering wheel angle (code line 86 in `model.py`)

Data are normalized in the model using Keras lambda layer (code line 17-19 in `model.py`)


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (code line 58, 64, 70, 76, 82 in `model.py`). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 46-48 in `model_execution.py`). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track without any crash. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model_execution.py` line 45).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving on the road with counter-clockwise (reverse) direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build extreme deep network as well as VGG.

My first step was to use a convolution neural network model similar to VGG. I thought this model might be appropriate because it repeats same convolution network for several times. Convolution network make my model good at learning image pattern.  
I prepared a few fully-connected layer at first which are connected between convolution layers and the final output layer.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 


To combat the overfitting, I modified the model so that dropout and batch normalization are introduced my model. Especially, dropout is effective to generalize my model.


The final step was to run the simulator to see how well the car was driving around track one. 
At beginning trials, the car fell the ground out of the truck without steering operation. 
I shrink my model and incremented layer step by step after the beginning trial. 
I found that serving multiple fully-connected network layer is preferred way to train my model effectively.
Only deep convolution network does not mean that the autonomous car can drive well on the road.  
I added layers to my model trial by trial.
I found that several fully-connected layers make the car motion stable, and convolution layer works well due to fully-connected layers

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road and crashing.

#### 2. Final Model Architecture

The final model architecture (defined `build_mymodel()` function in model.py) consisted of convolution neural network layers and it has been introduced on the previous section in this writeup.

Here is a visualization of the architecture which was generated by `plot_model()` function.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example movie of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to avoid attacking anything and leaving the road. These movies show how I got images where the car turning appropriate direction to avoid leaving the road, crashing wall, or falling on the ground.

![alt text][image3]


![alt text][image4]


Then I repeated this process on track two in order to get more data points.

After I got images with driving the car with the counter-clockwise direction, it will be helpful to train my model with different cases.

To augment the data sat, I also flipped images and angles thinking that this would let my model learn driving more accurately. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 22628 number of data points, where the half of data points are obtained by data augmentation.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I chose 10 as epoch number because loss value stop declining arond this value. I used an adam optimizer so that manually training the learning rate wasn't necessary.
