import pydicom 
import os.path as op 
import glob 
import numpy as np 
import imageio as iio 
from subprocess import run 
import shutil
import os 
import re

INDIR = 'MRI_png'
OUTIDR = 'MRI_png_processed'

crop = 40 
frame_range = (2/36, 28/36)

# def extract_subs():
#     dicoms = glob.glob(op.join('test_knees', 'dicom', '*.dcm'))
#     subs = [ int(op.split(d)[1][4:8]) for d in dicoms ]
#     return np.unique(subs)

# def make_png():

#     subs = extract_subs()

#     for sub in subs: 
#         frames = []
#         dicoms = sorted(glob.glob(op.join("test_knees", "dicom", "IMG-{:04d}*.dcm".format(sub))))
#         dcm_len = len(dicoms)
#         dcm_range = range(*[int(f * dcm_len) for f in frame_range])
#         for dcidx in dcm_range:
#             frames.append(pydicom.dcmread(dicoms[dcidx]).pixel_array)
        
#         img = np.stack(frames, axis=2)[crop:-crop, crop:-crop,...]
#         img = (255 * (img/img.max())).astype(np.uint8)
#         for idx,dcidx in enumerate(dcm_range):
#             outpath = op.join("test_knees", "png", "sub-{:03d}-img{:03d}.png".format(sub,dcidx))
#             iio.imsave(outpath, img[:,:,idx])

def make_masks(): 

    for sdir, _, files in os.walk(INDIR):
        for f in files: 
            if f.endswith('.json'):
                stub = f[:f.index('.json')]
                f = op.join(sdir, f)
                out = op.join(OUTIDR, stub)
                cmd = f"labelme_json_to_dataset {f} -o {out}"
                run(cmd, shell=True)
   
def load_img_data():
    pass

if __name__ == "__main__":
    
    make_masks()