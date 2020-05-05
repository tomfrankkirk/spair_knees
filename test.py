# Testing script for unet 

import numpy as np
from keras.models import load_model 
from image_generator import ImageGenerator
from train import image_label_paths, IMG_SHAPE, CLASS_DICT
import imageio as iio 
import matplotlib.pyplot as plt 

INDIR = 'MRI_png_processed'
OUTDIR = 'MRI_predicted'
TITLES = ['Predicted', 'Truth']


if __name__ == "__main__":
    
    generate = ImageGenerator(*image_label_paths(), IMG_SHAPE, 1, CLASS_DICT)
    unet = load_model('unet.h5')

    for images, labels in generate:
        predictions = unet.predict_on_batch(images)
        
        for img, pred, lbl in zip(images, predictions, labels):
            fig, axes = plt.subplots(1, 3, constrained_layout=True)
            for ax, title in zip(axes, ['blah'] + TITLES):
                pass 
