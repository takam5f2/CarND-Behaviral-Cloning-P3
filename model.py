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
    model.add(Lambda(lambda x: (x / 128.0) - 1.0,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    # model.add(Cropping2D(cropping=((60,20), (0,0)))) # including hyperparameter
    # VGG-like Neural Network
    # convolutional
    for i in range(2):
         model.add(Conv2D(6, (3, 3), strides=(2,2), padding='same'))
         model.add(BatchNormalization(epsilon=1e-07))
         model.add(Activation('relu'))
    # max pooling
    model.add(MaxPool2D(pool_size=(2,2)))

    # convolutional
    for i in range(2):
        model.add(Conv2D(6, (3, 3), strides=(2,2), padding='same'))
        model.add(BatchNormalization(epsilon=1e-07))
        model.add(Activation('relu'))
    # max pooling
    model.add(MaxPool2D(pool_size=(2,2)))

    # convolutional
    for i in range(3):
        model.add(Conv2D(24, (3, 3), strides=(1,1), padding='same'))
        model.add(BatchNormalization(epsilon=1e-07))
        model.add(Activation('relu'))
    # max pooling
    model.add(MaxPool2D(pool_size=(2,2)))

    # convolutional
    model.add(Conv2D(96, (3, 3), strides=(1,1), padding='same'))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    
    # flatten
    model.add(Flatten())

    # fully
    model.add(Dense(800))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(800))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(600)) 
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(300))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(120))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    
    return model
