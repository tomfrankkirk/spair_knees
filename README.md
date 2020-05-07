## Unet for segmenting vessels from axial knee SPAIR images

### Overview
The objective is to segment small vessles within the tibia as imaged on axial SPAIR-MR images of the knee. We have a small number of labelled images as ground truth (currently 28), so we are using data augmentation pretty aggressively. Fortunately all the scans are pretty similar so with conservtive settings I think this is valid. 

### Visualising the data 
Use the notebook `visualise.ipynb` to have a look at the augmented data and also the predicted output (futher down on the page). We have 4 classes: pure background, other tissue (yellow), tibia (green), and vessel (blue). We want to segment the vessels and ideally count them per slice. 

### Training the network
`dicom_to_png.py` extracts the raw DICOM scans and dumps them to png files. These were then annoted with `labelme` to produce json files. `extract_masks` binarises the masks and dumps them to png in the directory `MRI_png_processed`. `datasource.py` contains custom subclasses of `keras.utils.Sequential` that handle loading the image-mask pairs, converting them to one-hot labels in 4 classes, transforming and augmenting them according to the settings `EXPANSION_FACTOR` and `AUGMENT_ARGS`. The architecture of the network itself is given in `unet.py` and the training is handled by `train.py`. 

### Open questions 
1 Number of layers, which in turn fixes the number of filters at the bottom layer? Given the very fine nature of the details we want to segment, how could we tweak the network to better target these?

2 Currently there are no negative samples in the dataset (ie, patients with no vessels at all). Given how small the vessels are, it is doubtful this will significantly bias the training right now. 

3 Would it be better to target another objective? Ie, instead of learning to segment vessels, just train directly on the count of vessels per slice? I think this would require an enormous dataset to work. 
