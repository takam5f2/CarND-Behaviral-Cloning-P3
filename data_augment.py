# data augmentation
# copyright: Takayuki AKAMINE

from scipy import ndimage
from scipy.misc import imsave
import numpy as np

def data_augment(samples, img_path='../small_dataset/driving_log.csv'):
    new_samples = []
    for sample in samples:
        new_sample = ['', '', '', 0., 0., 0., 0.]
        src_name = img_path+sample[0].split('/')[-1]
        center_image = ndimage.imread(src_name)
        center_angle = float(sample[3])
        # flip
        flip_name = img_path+"flip_"+sample[0].split('/')[-1]
        flipped_image = np.fliplr(center_image)
        imsave(flip_name, flipped_image)
        flipped_angle = -center_angle
        new_sample[0] = flip_name
        new_sample[3] = flipped_angle

        new_samples.append(new_sample)
    samples.extend(new_samples)
    return samples
        

