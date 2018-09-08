"""
Generator is defined in this soruce code
This function is used for batch processing in machine learning
"""

# import cv2
import numpy as np
import sklearn
from scipy import ndimage

def generator(samples, batch_size=32, img_path='../small_dataset/IMG/'):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        batch_samples = sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            qbatch_samples = batch_samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in qbatch_samples:
                name = img_path+batch_sample[0].split('/')[-1]
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

