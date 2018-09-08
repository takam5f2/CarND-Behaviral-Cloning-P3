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
    # 2 convolutional layer: filter size=(3x3), depth = 6
    for i in range(2):
         model.add(Conv2D(6, (3, 3), strides=(2,2), padding='same'))
         model.add(BatchNormalization(epsilon=1e-07))
         model.add(Activation('relu'))
    # max pooling with pooling size = (2,2)
    model.add(MaxPool2D(pool_size=(2,2)))

    # 2 convolutional layer: filter size=(3x3), depth = 6
    for i in range(2):
        model.add(Conv2D(6, (3, 3), strides=(2,2), padding='same'))
        model.add(BatchNormalization(epsilon=1e-07))
        model.add(Activation('relu'))
    # max pooling with pooling size = (2,2)
    model.add(MaxPool2D(pool_size=(2,2)))

    # 3 convolutional layer: filter size=(3x3), depth = 24
    for i in range(3):
        model.add(Conv2D(24, (3, 3), strides=(1,1), padding='same'))
        model.add(BatchNormalization(epsilon=1e-07))
        model.add(Activation('relu'))
    # max pooling with pooling size = (2,2)
    model.add(MaxPool2D(pool_size=(2,2)))

    # 1 convolutional layer: filter size=(3x3), depth = 96
    model.add(Conv2D(96, (3, 3), strides=(1,1), padding='same'))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    
    # flatten
    model.add(Flatten())

    # fully connected: output size=800
    model.add(Dense(800))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fully connected: output size=800
    model.add(Dense(800))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # fully connected: output size=600
    model.add(Dense(600)) 
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fully connected: output size=300
    model.add(Dense(300))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fully connected: output size=120
    model.add(Dense(120))
    model.add(BatchNormalization(epsilon=1e-07))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fully connected: output size=1
    # output value is used for steering wheel angle.
    model.add(Dense(1))
    
    return model
