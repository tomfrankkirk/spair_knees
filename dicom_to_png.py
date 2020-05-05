import pydicom
import os.path as op 
import glob
import numpy as np 
import imageio as iio 
import os

indir = '/Volumes/UNTITLED/MRI_XR_sorted'
outdir = '/Volumes/UNTITLED/MRI_png'

sdirs = glob.glob(f'{indir}/*/MRI*')

for sidx,sdir in enumerate(sdirs): 

    dirname = op.split(sdir)[1]
    odir = op.join(outdir, 'sub-{:03d}'.format(sidx+1))
    os.makedirs(odir, exist_ok=True)

    frames = []
    dicoms = sorted(glob.glob(op.join(sdir, '*AX PD_SPAIR*.dcm')))
    if not dicoms: continue 
    
    for dcm in dicoms:
        frames.append(pydicom.dcmread(dcm).pixel_array)

    img = np.stack(frames, axis=2)
    img = (255 * (img / img.max())).astype(np.uint8)

    for fidx in range(img.shape[-1]):
        outpath = op.join(odir, "sub-{:03d}_img-{:03d}.png".format(sidx+1,fidx))
        iio.imsave(outpath, img[:,:,fidx])


# 'RT_AX_PD_SPAIR_MR*.dcm '