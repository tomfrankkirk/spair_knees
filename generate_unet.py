# This code is produced for DTC teaching use.
# Author : Weidi Xie (weidi@robots.ox.ac.uk)

from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, Reshape, Activation, Dropout

import numpy as np
weight_decay = 1e-4


def get_unet(input_dim, output_dim, num_output_classes):

    # input 
    # Block 1
    img_input = Input(shape=(*input_dim, 1), name='image_slice')
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x1)
    ds1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)

    # Block 2
    # input is of size : 128 x 128 x 64
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(ds1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x2)
    ds2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)

    # Block 3
    # input is of size : 64 x 64 x 128
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(ds2)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x3)
    ds3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)

    # Block 4
    # input is of size : 32 x 32 x 256
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(ds3)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x4)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x4)

    # Upsampling 1.
    us1 = concatenate([UpSampling2D(size=(2,2))(x4), x3])
    x5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us1_conv1', trainable=True)(us1)
    x5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us1_conv2', trainable=True)(x5)

    # Upsampling 2.
    us2 = concatenate([UpSampling2D(size=(2, 2))(x5), x2])
    x6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us2_conv1', trainable=True)(us2)
    x6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(x6)

	# Upsampling 3.
    us3 = concatenate([UpSampling2D(size=(2, 2))(x6), x1])
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us3_conv1', trainable=True)(us3)
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(x7)

    dense_prediction = Conv2D(
                             num_output_classes,
                             (3,3),
                             padding='same',
                             activation='linear',
                             kernel_initializer='orthogonal',
                             kernel_regularizer=l2(weight_decay),
                             bias_regularizer=l2(weight_decay))(x7)

    map_flatten = Reshape((np.product(output_dim[:2]), num_output_classes))(dense_prediction)
    map_activation = Activation(activation='sigmoid')(map_flatten)
    segmentation = Reshape((output_dim[0], output_dim[1], num_output_classes), name = 'segmentation')(map_activation)

    model = Model(inputs = img_input, outputs = segmentation)

    model.compile(optimizer = Adam(1e-4), loss = 'binary_crossentropy', metrics=['categorical_accuracy'])

    return model


