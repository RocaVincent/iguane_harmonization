from subprocess import run
from tempfile import gettempdir
from uuid import uuid4
import nibabel as nib
import numpy as np
from harmonization.model_architectures import Generator
from tensorflow import convert_to_tensor
from tensorflow.config import list_physical_devices
from HD_BET.run import run_hd_bet
from os import remove


if len(list_physical_devices('GPU'))>0:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    

TMP_DIR = gettempdir()
TEMPLATE_PATH = 'preprocessing/MNI152_T1_1mm_brain.nii.gz'

def tmp_unique_path(extension='.nii.gz'):
    return f"{TMP_DIR}/{uuid4().hex}{extension}"

def run_cmd(cmd):
    run(cmd.split(' '), capture_output=False)
    
def indices_crop(data):
    # count number of zeros
    d1_1=0
    while d1_1<data.shape[0] and np.count_nonzero(data[d1_1,:,:])==0: d1_1+=1
    d1_2=0
    while d1_2<data.shape[0] and np.count_nonzero(data[-d1_2-1,:,:])==0: d1_2+=1
    d2_1=0
    while d2_1<data.shape[1] and np.count_nonzero(data[:,d2_1,:])==0: d2_1+=1
    d2_2=0
    while d2_2<data.shape[1] and np.count_nonzero(data[:,-d2_2-1,:])==0: d2_2+=1
    d3_1=0
    while d3_1<data.shape[1] and np.count_nonzero(data[:,:,d3_1])==0: d3_1+=1
    d3_2=0
    while d3_2<data.shape[1] and np.count_nonzero(data[:,:,-d3_2-1])==0: d3_2+=1
    
    # determine cropping
    if d1_1+d1_2 >= 22:
        if d1_1<11: xmin = d1_1
        elif d1_2<11: xmin = 182-160-d1_2
        else: xmin = 11
        xsize = 160
    else: xmin, xsize = 3,176
        
    if d2_1+d2_2 >= 26:
        if d2_1<13: ymin = d2_1
        elif d2_2<13: ymin = 218-192-d2_2
        else: ymin = 13
        ysize = 192
    else: ymin, ysize = 5,208
    
    if d3_1+d3_2 >= 22:
        if d3_1<11: zmin = d3_1
        elif d3_2<11: zmin = 182-160-d3_2
        else: zmin = 11
        zsize = 160
    else: zmin, zsize = 3,176
    return xmin, xsize, ymin, ysize, zmin, zsize


def run_singleproc(in_mri, out_mri, hd_bet_cpu):
    reorient_path = tmp_unique_path()
    run_cmd(f"fslreorient2std {in_mri} {reorient_path}")
    
    brain_native = tmp_unique_path()
    if hd_bet_cpu: run_hd_bet(reorient_path, brain_native, bet=True, device='cpu', mode='fast', do_tta=False)
    else: run_hd_bet(reorient_path, brain_native, bet=True)
    mask_native = brain_native[:-7]+'_mask.nii.gz'
    
    n4native = tmp_unique_path()
    run_cmd(f"N4BiasFieldCorrection -i {brain_native} -x {mask_native} -o {n4native}")
    
    n4mni = tmp_unique_path()
    mni_mat = tmp_unique_path('.mat')
    run_cmd(f"flirt -in {n4native} -ref {TEMPLATE_PATH} -omat {mni_mat} -interp trilinear -dof 6 -out {n4mni}")
    
    mask_mni = tmp_unique_path()
    run_cmd(f"flirt -in {mask_native} -ref {TEMPLATE_PATH} -out {mask_mni} -init {mni_mat} -applyxfm -interp nearestneighbour")
    
    median_mni = tmp_unique_path()
    mri = nib.load(n4mni)
    data = mri.get_fdata()
    mask = nib.load(mask_mni).get_fdata() > 0
    med = np.median(data[mask])
    data /= med 
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    nib.save(mri, median_mni)
    
    xmin, xsize, ymin, ysize, zmin, zsize = indices_crop(data)
    median_crop = tmp_unique_path()
    run_cmd(f"fslroi {median_mni} {median_crop} {xmin} {xsize} {ymin} {ysize} {zmin} {zsize}")
    
    generator = Generator()
    generator.load_weights('harmonization/iguane_weights.h5')
    mri = nib.load(median_crop)
    data = mri.get_fdata()
    mask = data>0
    data -= 1
    data[~mask] = 0
    data = np.expand_dims(data, axis=(0,4))
    t = convert_to_tensor(data, dtype='float32')
    data = generator(t, training=False).numpy().squeeze()
    data = (data+1) * 500
    data[~mask] = 0
    data = np.maximum(data, 0)
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    nib.save(mri, out_mri)
    
    remove(reorient_path)
    remove(brain_native)
    remove(mask_native)
    remove(n4native)
    remove(n4mni)
    remove(mni_mat)
    remove(mask_mni)
    remove(median_mni)
    remove(median_crop)