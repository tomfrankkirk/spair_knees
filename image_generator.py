import keras 
import os.path as op
import glob 
import numpy as np 
import imageio as iio 
from skimage.transform import resize
from pdb import set_trace

IMG_SHAPE = (560, 560)
N_CLASSES = 3
TISSUE_MAP = {
    'Mark': 1, 
    'vessel': 1, 
    'bone': 2, 
    'Tibia': 2, 
    '_background_': 0 
}


class ImageGenerator(keras.utils.Sequence):

    def __init__(self, images, labels, batch_size):
        self.batch_size = batch_size
        self.image_paths = images
        self.label_paths = labels
        self.on_epoch_end()

    @property
    def indices(self):
        return list(range(len(self.image_paths)))

    def _generate_batch(self, index_list):
        images = np.empty((self.batch_size, *IMG_SHAPE, 1), dtype=np.float32)
        labels = np.zeros((self.batch_size, *IMG_SHAPE), dtype=np.uint8)

        for iidx,idx in enumerate(index_list):
            img = iio.imread(self.image_paths[idx])
            img = resize(img, IMG_SHAPE, anti_aliasing=True) / 255
            images[iidx,:,:,0] = (img - img.mean()) / img.std()

            lbl = iio.imread(self.label_paths[idx])
            lbl = resize(lbl, IMG_SHAPE, preserve_range=True) / 255 
            lbl_names = open(self.label_paths[idx][:-4]+'_names.txt', 'r').readlines()
            lbl_flat = np.zeros(IMG_SHAPE, dtype=np.uint8)

            for idx,(name,channel) in enumerate(zip(lbl_names[1:], [0,1,2])):
                mask = (lbl[:,:,idx] > 0.25)
                lbl_flat[mask] = TISSUE_MAP[name.strip()]
            labels[iidx,:,:] = lbl_flat 

        labels = keras.utils.to_categorical(labels, num_classes=N_CLASSES)
        return images, labels

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return self._generate_batch(inds)
        