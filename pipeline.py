from subprocess import run
from tempfile import gettempdir, mkdtemp
from uuid import uuid4
import nibabel as nib
import numpy as np
from harmonization.model_architectures import Generator
from tensorflow import convert_to_tensor
from tensorflow.config import list_physical_devices
from HD_BET.run import run_hd_bet
from os import remove, symlink
from shutil import rmtree
from multiprocessing import Pool



if len(list_physical_devices('GPU'))>0:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    

TMP_DIR = gettempdir()
TEMPLATE_PATH = 'preprocessing/MNI152_T1_1mm_brain.nii.gz'

def tmp_unique_path(extension='.nii.gz'):
    return f"{TMP_DIR}/{uuid4().hex}{extension}"

def run_cmd(cmd):
    return run(cmd.split(' '), capture_output=True)
    
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
    returned = run_cmd(f"fslreorient2std {in_mri} {reorient_path}")
    if returned.stderr:
        print(f"Problem fslreorient2std : {returned.stderr.decode('utf-8')}")
        return False
    
    brain_native = tmp_unique_path()
    if hd_bet_cpu: run_hd_bet(reorient_path, brain_native, bet=True, device='cpu', mode='fast', do_tta=False)
    else: run_hd_bet(reorient_path, brain_native, bet=True)
    mask_native = brain_native[:-7]+'_mask.nii.gz'
    
    n4native = tmp_unique_path()
    returned = run_cmd(f"N4BiasFieldCorrection -i {brain_native} -x {mask_native} -o {n4native}")
    if returned.returncode != 0:
        print("Problem with N4BiasFieldCorrection")
        return False
    
    n4mni = tmp_unique_path()
    mni_mat = tmp_unique_path('.mat')
    returned = run_cmd(f"flirt -in {n4native} -ref {TEMPLATE_PATH} -omat {mni_mat} -interp trilinear -dof 6 -out {n4mni}")
    if returned.stderr:
        print(f"Problem flirt : {returned.stderr.decode('utf-8')}")
        return False
    
    mask_mni = tmp_unique_path()
    returned = run_cmd(f"flirt -in {mask_native} -ref {TEMPLATE_PATH} -out {mask_mni} -init {mni_mat} -applyxfm -interp nearestneighbour")
    if returned.stderr:
        print(f"Problem flirt : {returned.stderr.decode('utf-8')}")
        return False
    
    median_mni = tmp_unique_path()
    mri = nib.load(n4mni)
    data = mri.get_fdata()
    mask = nib.load(mask_mni).get_fdata() > 0
    med = np.median(data[mask])
    data = data/med
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    nib.save(mri, median_mni)
    
    xmin, xsize, ymin, ysize, zmin, zsize = indices_crop(data)
    median_crop = tmp_unique_path()
    returned = run_cmd(f"fslroi {median_mni} {median_crop} {xmin} {xsize} {ymin} {ysize} {zmin} {zsize}")
    if returned.stderr:
        print(f"Problem fslroi : {returned.stderr.decode('utf-8')}")
        return False
    
    generator = Generator()
    generator.load_weights('harmonization/iguane_weights.h5')
    mri = nib.load(median_crop)
    data = mri.get_fdata()-1
    mask = data>-1
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
    
    
    
    
def reorient(directory, id_):
    returned = run_cmd(f"fslreorient2std {directory}/base_{id_}.nii.gz {directory}/reorient_{id_}.nii.gz")
    if returned.stderr:
        print(f"Problem fslreorient2std for entry number {id_} : {returned.stderr.decode('utf-8')}")
        return False
    return True

def n4_flirt_median_crop(directory, id_):
    returned = run_cmd(f"N4BiasFieldCorrection -i {directory}/brainNative_{id_}.nii.gz -x {directory}/brainNative_{id_}_mask.nii.gz -o {directory}/n4native_{id_}.nii.gz")
    if returned.returncode!=0:
        print(f"Problem N4BiasField for entry number {id_}")
        return False
    returned = run_cmd(f"flirt -in {directory}/n4native_{id_}.nii.gz -ref {TEMPLATE_PATH} -omat {directory}/flirtMat_{id_}.mat -interp trilinear -dof 6 -out {directory}/n4mni_{id_}.nii.gz")
    if returned.stderr:
        print(f"Problem flirt for entry number {id_} : {returned.stderr.decode('utf-8')}")
        return False
    returned = run_cmd(f"flirt -in {directory}/brainNative_{id_}_mask.nii.gz -ref {TEMPLATE_PATH} -out {directory}/maskMni_{id_}.nii.gz -init {directory}/flirtMat_{id_}.mat -applyxfm -interp nearestneighbour")
    if returned.stderr:
        print(f"Problem flirt apply transform for entry number {id_} : {returned.stderr.decode('utf-8')}")
        return False

    mri = nib.load(f"{directory}/n4mni_{id_}.nii.gz")
    data = mri.get_fdata()
    mask = nib.load(f"{directory}/maskMni_{id_}.nii.gz").get_fdata() > 0
    median = np.median(data[mask])
    data = data/median
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    nib.save(mri, f"{directory}/medianNorm_{id_}.nii.gz")

    xmin, xsize, ymin, ysize, zmin, zsize = indices_crop(data)
    returned = run_cmd(f"fslroi {directory}/medianNorm_{id_}.nii.gz {directory}/crop_{id_}.nii.gz {xmin} {xsize} {ymin} {ysize} {zmin} {zsize}")
    if returned.stderr:
        print(f"Problem fslroi apply transform for entry number {id_} : {returned.stderr.decode('utf-8')}")
        return False
    return True
    
def run_multiproc(in_paths, out_paths, n_procs, hd_bet_cpu):
    tmp_dir = mkdtemp()
    ids = range(1, len(in_paths)+1)
    bases = [f"{tmp_dir}/base_{id_}.nii.gz" for id_ in ids]
    for i,id_ in enumerate(ids): symlink(in_paths[i], f"{tmp_dir}/base_{id_}.nii.gz")
    
    print("reorientation of MR images...")
    with Pool(n_procs) as pool:
        flags = pool.starmap(reorient, [(tmp_dir,id_) for id_ in ids])
    ids = [ids[i] for i in range(len(ids)) if flags[i]]
    
    print('\nHD-BET brain extractions...')
    hd_bet_in = [f"{tmp_dir}/reorient_{id_}.nii.gz" for id_ in ids]
    hd_bet_out = [f"{tmp_dir}/brainNative_{id_}.nii.gz" for id_ in ids]
    if hd_bet_cpu: run_hd_bet(hd_bet_in, hd_bet_out, bet=True, device='cpu', mode='fast', do_tta=False)
    else: run_hd_bet(hd_bet_in, hd_bet_out, bet=True)
    
    print('\nBias field correction, registration, median normalization and cropping...')
    with Pool(n_procs) as pool:
        flags = pool.starmap(n4_flirt_median_crop, [(tmp_dir, id_) for id_ in ids])
    ids = [ids[i] for i in range(len(ids)) if flags[i]]
    
    print('IGUANe harmonization...')
    generator = Generator()
    generator.load_weights('harmonization/iguane_weights.h5')
    for i,id_ in enumerate(ids, start=1):
        print(f"\tHarmonizing image {i}/{len(ids)}...", end='\r')
        mri = nib.load(f"{tmp_dir}/crop_{id_}.nii.gz")
        data = mri.get_fdata()-1
        mask = data>-1
        data[~mask] = 0
        data = np.expand_dims(data, axis=(0,4))
        t = convert_to_tensor(data, dtype='float32')
        data = generator(t, training=False).numpy().squeeze()
        data = (data+1) * 500
        data[~mask] = 0
        data = np.maximum(data, 0)
        mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
        try: nib.save(mri, out_paths[id_-1])
        except FileNotFoundError:
            print(f"Problem saving {out_paths[id_-1]}")
        
    print('Deletion of temporary files...                     ')
    rmtree(tmp_dir)