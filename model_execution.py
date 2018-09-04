# model_execution.py
# this source code performs training, validation
# copyright: Takayuki AKAMINE

import os
import csv

samples = []
with open('../sample_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


from generator import generator
# compile and train the model using the generator function
img_path = '../sample_data/IMG/'
train_generator = generator(train_samples, batch_size=32, img_path=img_path)
validation_generator = generator(validation_samples, batch_size=32, img_path=img_path)

# data augment

ch, row, col = 3, 160, 320  # Trimmed image format

# model 
from keras.models import Sequential
from model import build_mymodel
model = Sequential()
model = build_mymodel(model, ch, row, col)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples),
                    validation_data=validation_generator,
                      nb_val_samples = len(validation_samples), nb_epoch=25)
model.save('model.h5')
