import imageio as iio 
import os.path as op 
import numpy as np 
from generate_unet import get_unet
from skimage.transform import resize 
from datasource import ImageGenerator, N_CLASSES, IMG_SHAPE, augmented_data, image_label_paths
from datasource import AugmentedGenerator
import tensorflow as tf 
import os 
import math 

SPLIT = 0.8 
INDIR = 'MRI_png_processed'    
BATCH_SIZE = 4
EPOCHS = 20
EXPANSION_FACTOR = 1
# CLASS_WEIGHTS = { 
#     0: 0.2, 
#     1: 1,  
#     2: 0.5, 
#     3: 0.5 }

AUGMENT_ARGS = dict( 
    horizontal_flip=True, 
    vertical_flip=True, 
    zoom_range=0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1)



if __name__ == "__main__":

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=14, 
                        inter_op_parallelism_threads=2, 
                        allow_soft_placement=True,
                        device_count = {'CPU': 16})

    session = tf.compat.v1.Session(config=config)
    # tf.debugging.set_log_device_placement(True)

    # Split into test and validation sets 
    images, labels = image_label_paths(INDIR)
    partition = int(len(images) * SPLIT)
    images_train = images[:partition]
    labels_train = labels[:partition]
    images_val = images[partition:]
    labels_val = labels[partition:]

    # intitiailse generators on each set of data
    train_gen = ImageGenerator(images_train, labels_train, BATCH_SIZE)
    val_gen = ImageGenerator(images_val, labels_val, BATCH_SIZE) 
    augmented_train = AugmentedGenerator(train_gen, EXPANSION_FACTOR, AUGMENT_ARGS)
    augmented_val = AugmentedGenerator(val_gen, EXPANSION_FACTOR, AUGMENT_ARGS)

    unet = get_unet(IMG_SHAPE, IMG_SHAPE, N_CLASSES)
    unet.summary()

    print(len(images), "samples, in", len(train_gen), "batches")
    train_steps = len(train_gen) * EXPANSION_FACTOR
    print("{} * {} = {} steps per epoch".format(len(train_gen), EXPANSION_FACTOR, train_steps))
    val_steps = len(val_gen) * EXPANSION_FACTOR

    unet.fit_generator(generator=augmented_train, validation_data=augmented_val,
        epochs=EPOCHS, verbose=1, shuffle=True, steps_per_epoch=train_steps, validation_steps=val_steps)

    unet.save('unet.h5')
