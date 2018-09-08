# data augmentation
# copyright: Takayuki AKAMINE

from scipy import ndimage
from scipy.misc import imsave
import numpy as np

def data_augment(samples, img_path='../small_dataset/driving_log.csv'):
    new_samples = [] # array for augment data.
    for sample in samples:
        new_sample = ['', '', '', 0., 0., 0., 0.]
        # open origianl images
        src_name = img_path+sample[0].split('/')[-1]
        center_image = ndimage.imread(src_name)
        center_angle = float(sample[3])
        # flip
        flipped_image = np.fliplr(center_image)
        flipped_angle = -center_angle
        # save flipped image to file named with flip_name
        flip_name = img_path+"flip_"+sample[0].split('/')[-1]
        imsave(flip_name, flipped_image)
        # add augment file name and steering wheel angle.
        new_sample[0] = flip_name
        new_sample[3] = flipped_angle
        new_samples.append(new_sample)
    samples.extend(new_samples)
    return samples
        

