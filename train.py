# Brain segmentation U-network
# Based on the work of 
# Tom Kirk, 5/3/18

# Dependencises
import imageio as iio 
import os.path as op 
import glob 
import numpy as np 
from generate_unet import get_unet
from keras.utils.np_utils import to_categorical
from skimage.transform import resize 
import keras 
from image_generator import ImageGenerator, N_CLASSES, IMG_SHAPE
import tensorflow as tf 
import os 

SPLIT = 0.8 
INDIR = 'MRI_png_processed'    
BATCH_SIZE = 10

def image_label_paths():

    images = [] 
    for sdir,_,files in os.walk(INDIR):
        for f in files:
            if f == 'img.png':
                images.append(op.join(sdir, f))

    images = sorted(images)
    labels = [ op.join(op.dirname(f), 'label.png') for f in images ]
    return images, labels 

if __name__ == "__main__":

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=14, 
                        inter_op_parallelism_threads=2, 
                        allow_soft_placement=True,
                        device_count = {'CPU': 16})

    session = tf.compat.v1.Session(config=config)
    # tf.debugging.set_log_device_placement(True)

    images, labels = image_label_paths()
    partition = int(len(images) * SPLIT)
    images_train = images[:partition]
    labels_train = labels[:partition]
    images_test = images[partition:]
    labels_test = labels[partition:]

    train_gen = ImageGenerator(images_train, labels_train, BATCH_SIZE) 
    test_gen = ImageGenerator(images_test, labels_test, BATCH_SIZE)

    # Prepare network
    unet = get_unet(IMG_SHAPE, IMG_SHAPE, N_CLASSES)
    unet.summary()

    # use_multiprocessing=True, workers=8
    unet.fit_generator(generator=train_gen, validation_data=test_gen, 
        epochs=10, verbose=1)
    unet.save('unet.h5')