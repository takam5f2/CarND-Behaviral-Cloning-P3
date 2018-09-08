# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, I used deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested my model using Keras. The model outputs a steering angle to an autonomous vehicle.

A simulator was provided, where I steer a car around a track for data collection. I used image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

I created a detailed writeup of the project. Check out the [writeup](https://github.com/takam5f2/CarND-Behaviral-Cloning-P3/blob/master/writeup.md).

To meet specifications, the project will require submitting five files: 
* model.py (script for defining the model)
* model_execution.py (script to read data, execute data augmentation, train my model with validation data)
* generator.py (script for generating image data according to batch size given as argument)
* data_augment.py (script for defining function for executing data augmentation with flip function)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* autonomous_driving.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)
* a report writeup file (either markdown or pdf)


This README file describes how to output the video in the "Details About Files In This Directory" section.


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

