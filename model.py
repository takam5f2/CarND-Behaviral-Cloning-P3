# model.py
# this source code includes model for control vehicle steering wheel angle
# with machine learning system.
# copyright: Takayuki AKAMINE

# model 
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization

def build_mymodel(model, ch=3, row=160, col=320):
    

    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70,25), (0,0)))) # including hyperparameter
    # VGG
    # convolutional
    for i in range(2):
         model.add(Conv2D(6, (3, 3), strides=(1,1), padding='same'))
         model.add(BatchNormalization(epsilon=1e-07))
         model.add(Activation('relu'))
    # max pooling
    model.add(MaxPool2D(pool_size=(2,2)))

    # convolutional
    for i in range(2):
        model.add(Conv2D(12, (3, 3), strides=(1,1), padding='same'))
        model.add(BatchNormalization(epsilon=1e-07))
        model.add(Activation('relu'))
    # max pooling
    model.add(MaxPool2D(pool_size=(2,2)))

    
    # flatten
    model.add(Flatten(input_shape=(row, col, ch)))

    # fully
    model.add(Dense(1200))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(480))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(120))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(48))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    
    return model
