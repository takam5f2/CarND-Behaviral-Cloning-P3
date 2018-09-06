# model_execution.py
# this source code performs training, validation
# copyright: Takayuki AKAMINE

import os
import csv
from keras.utils import plot_model

samples = []
data_path = '../sample_data/'
img_path = data_path + 'IMG/'
with open(data_path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# data augment
from data_augment import data_augment
train_samples = data_augment(train_samples, img_path=img_path)

from generator import generator
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, img_path=img_path)
validation_generator = generator(validation_samples, batch_size=32, img_path=img_path)

ch, row, col = 3, 160, 320  # Trimmed image format

# model 
from keras.models import Sequential
from model import build_mymodel
model = Sequential()
model = build_mymodel(model, ch, row, col)

plot_model(model, to_file='mymodel.png')

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)/8,
                    validation_data=validation_generator,
                    validation_steps = len(validation_samples)/8, epochs=10)
model.save('model.h5')