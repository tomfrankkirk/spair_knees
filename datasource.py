import keras 
from keras.preprocessing.image import ImageDataGenerator
import os.path as op
import os 
import numpy as np 
import imageio as iio 
from skimage.transform import resize
from pdb import set_trace
from scipy.spatial import Delaunay

CROP = 40 # Crop in by this many voxels either side 
IMG_SHAPE = (480, 480) # For the first pass of training 
SUB_SHAPE = (190,240) # For the second pass of training 
TISSUE_MAP = {
    'Mark': 1, 
    'vessel': 1, 
    'bone': 2, 
    'Tibia': 2, 
    'other_tissue': 3,
    '_background_': 0 
}
N_CLASSES = np.unique(list(TISSUE_MAP.values())).size


def image_label_paths(indir):
    images = [] 
    for sdir,_,files in os.walk(indir):
        for f in files:
            if f == 'img.png':
                images.append(op.join(sdir, f))

    images = sorted(images)
    labels = [ op.join(op.dirname(f), 'label.png') for f in images ]
    return images, labels 


class AugmentedGenerator(keras.utils.Sequence):
    
    def __init__(self, data_generator, expansion_factor, augment_args):
        self.batch_size = data_generator.batch_size
        self.expansion_factor = expansion_factor
        self.data_generator = data_generator
        self.augment_args = augment_args
        self.seeds = np.random.randint(10e6, size=len(self))

    def __len__(self):
        return len(self.data_generator) * self.expansion_factor

    def __getitem__(self, index):
        transformer = ImageDataGenerator(**self.augment_args)
        seed = self.seeds[index]

        idx = (index // self.expansion_factor)
        if idx < len(self.data_generator):
            images, labels = self.data_generator.__getitem__(idx)
            aug_images = transformer.flow(images, batch_size=self.batch_size, seed=seed)
            aug_labels = transformer.flow(labels, batch_size=self.batch_size, seed=seed)
            return (next(aug_images), next(aug_labels).round())
        else: 
            raise RuntimeError("Shouldn't get here")
    

def augmented_data(data_generator, expansion_factor, augment_args):
    batch_size = data_generator.batch_size

    transformer = ImageDataGenerator(**augment_args)
    seed = np.random.randint(10e6)

    for images, labels in iter(data_generator):
        aug_images = transformer.flow(images, batch_size=batch_size, seed=seed)
        aug_labels = transformer.flow(labels, batch_size=batch_size, seed=seed)
        for _, aimg, albl in zip(range(expansion_factor), aug_images, aug_labels):
            yield(aimg, albl.round())


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
            img = iio.imread(self.image_paths[idx])[CROP:-CROP, CROP:-CROP]
            img = resize(img, IMG_SHAPE, anti_aliasing=True)
            img = (img - img.mean()) / img.std()
            images[iidx,:,:,0] = img # + (-1 * img.min())

            lbl = iio.imread(self.label_paths[idx])[CROP:-CROP, CROP:-CROP, :]
            lbl = resize(lbl, IMG_SHAPE, preserve_range=True) / 255 
            lbl_names = open(self.label_paths[idx][:-4]+'_names.txt', 'r').readlines()
            lbl_flat = np.zeros(IMG_SHAPE, dtype=np.uint8)

            # Pixels that have label 0 and an intensity above minimum - reclassify as
            # other tissue 
            # set_trace()
            threshold = (img.max() - img.min()) * 0.025 + img.min()
            mask = (lbl_flat == 0) & (img > threshold)
            lbl_flat[mask] = TISSUE_MAP['other_tissue']

            for idx,(name,channel) in enumerate(zip(lbl_names[1:], [0,1,2])):
                mask = (lbl[...,idx] > 0.25)
                lbl_flat[mask] = TISSUE_MAP[name.strip()]
            labels[iidx,...] = lbl_flat

        labels = keras.utils.to_categorical(labels, num_classes=N_CLASSES)
        assert (labels > 0).sum(-1).max() == 1, "multiple labels for pixel"
        return images[:iidx+1,...], labels[:iidx+1,...]

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        return self._generate_batch(inds)
    

class BoneGenerator(ImageGenerator):

    def __init__(self, *args):
        super().__init__(*args)

    def _generate_batch(self, index_list):
        images = np.empty((self.batch_size, *IMG_SHAPE, 1), dtype=np.float32)
        labels = np.zeros((self.batch_size, *IMG_SHAPE), dtype=np.uint8)

        xy_grid = np.stack(np.meshgrid(*[range(s) for s in IMG_SHAPE], 
                            indexing='ij'), axis=-1).reshape(-1,2)

        for iidx,idx in enumerate(index_list):
            img = iio.imread(self.image_paths[idx])[CROP:-CROP, CROP:-CROP]
            img = resize(img, IMG_SHAPE, anti_aliasing=True)
            img = (img - img.mean()) / img.std()
            images[iidx,:,:,0] = img # + (-1 * img.min())

            lbl = iio.imread(self.label_paths[idx])[CROP:-CROP, CROP:-CROP, :]
            lbl = resize(lbl, IMG_SHAPE, preserve_range=True) / 255 
            lbl_names = open(self.label_paths[idx][:-4]+'_names.txt', 'r').readlines()
            lbl_names = [ l.strip() for l in lbl_names ]

            # Binary segmentation of bone vs not bone, then expanded via convhull 
            for idx,(name,channel) in enumerate(zip(lbl_names[1:], [0,1,2])):
                if (name == 'bone') or (name == 'Tibia'):
                    bone_xy = np.argwhere(lbl[...,idx] > 0.25)
                    bone_hull = Delaunay(bone_xy)
                    bone_mask = (bone_hull.find_simplex(xy_grid) >= 0)
                    labels[iidx,...] = bone_mask.reshape(IMG_SHAPE)

        labels = labels.reshape(*labels.shape, 1)
        return images[:iidx+1,...], labels[:iidx+1,...]

class VesselGenerator(ImageGenerator):

    def __init__(self, *args):
        super().__init__(*args)

    def _generate_batch(self, index_list):
        images = np.empty((self.batch_size, *SUB_SHAPE, 1), dtype=np.float32)
        labels = np.zeros((self.batch_size, *SUB_SHAPE), dtype=np.uint8)

        for iidx,idx in enumerate(index_list):

            img = iio.imread(self.image_paths[idx])
            img = (img - img.mean()) / img.std()

            lbl = iio.imread(self.label_paths[idx]) / 255 
            lbl_names = open(self.label_paths[idx][:-4]+'_names.txt', 'r').readlines()
            lbl_names = [ l.strip() for l in lbl_names ]

            xy_grid = np.stack(np.meshgrid(*[range(s) for s in img.shape], 
                    indexing='ij'), axis=-1).reshape(-1,2)

            # Extract bone vs background only. 
            for idx,(name,channel) in enumerate(zip(lbl_names[1:], [0,1,2])):
                if (name == 'bone') or (name == 'Tibia'):
                    bone_xy = np.argwhere(lbl[...,idx] > 0.25)
                    bone_hull = Delaunay(bone_xy)
                    bone_mask = (bone_hull.find_simplex(xy_grid) >= 0)
                    bone_mask = bone_mask.reshape(img.shape)

            # We now have a mask of bone vs not bone 
            # Extract the bounding box, crop the image and label down to this size 
            bbox = np.vstack((bone_xy.min(0), bone_xy.max(0)))
            img_bbox = img[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1]]
            lbl_bbox = lbl[bbox[0,0]:bbox[1,0], bbox[0,1]:bbox[1,1]]

            # binary mask for vessel 
            for idx,(name,channel) in enumerate(zip(lbl_names[1:], [0,1,2])):
                if (name == 'Mark') or (name == 'vessel'):
                    vessel_mask = (lbl_bbox[...,idx] > 0.1)

            # resize to standard shape, rebinarise 
            labels[iidx,...] = (resize(vessel_mask, SUB_SHAPE) > 0.01)
            images[iidx,...,0] = resize(img_bbox, SUB_SHAPE, anti_aliasing=True)

        labels = labels.reshape(*labels.shape, 1)
        return images[:iidx+1,...], labels[:iidx+1,...]