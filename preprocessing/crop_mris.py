import numpy as np
from nibabel import load as load_mri
from multiprocessing import Pool
from subprocess import run


"""
INPUTS to define:
- ref_paths: list of MRI paths that will be used to determine the number of slices to crop on each side.
- list_entries: list of entries, MR images to crop. Each entry is a liste of tuples with (i) path of the MR image to crop and (ii) destination path for the cropped image.
"""


ref_paths = None # to define
list_entries = None # to define
N_PROCS = None # to define


def indicesCrop(mri_path):
    mri = load_mri(mri_path).get_fdata()
    d1_1=0
    while d1_1<mri.shape[0] and np.count_nonzero(mri[d1_1,:,:])==0: d1_1+=1
    d1_2=0
    while d1_2<mri.shape[0] and np.count_nonzero(mri[-d1_2-1,:,:])==0: d1_2+=1
    d2_1=0
    while d2_1<mri.shape[1] and np.count_nonzero(mri[:,d2_1,:])==0: d2_1+=1
    d2_2=0
    while d2_2<mri.shape[1] and np.count_nonzero(mri[:,-d2_2-1,:])==0: d2_2+=1
    d3_1=0
    while d3_1<mri.shape[1] and np.count_nonzero(mri[:,:,d3_1])==0: d3_1+=1
    d3_2=0
    while d3_2<mri.shape[1] and np.count_nonzero(mri[:,:,-d3_2-1])==0: d3_2+=1
    return d1_1,d1_2,d2_1,d2_2,d3_1,d3_2

def process_entry(ref_path,entries_sub):
    print(f"Process IRM {ref_path}")
    d1_1,d1_2,d2_1,d2_2,d3_1,d3_2 = indicesCrop(ref_path)
    if d1_1+d1_2<22:
        print(f"PROBLEM with d1 of {ref_path}, nzeros < 22 = {d1_1+d1_2}")
        return
    if d2_1+d2_2<26:
        print(f"PROBLEM with d2 of {ref_path}, nzeros < 26 = {d2_1+d2_2}")
        return
    if d3_1+d3_2<22:
        print(f"PROBLEM with d3 of {ref_path}, nzeros < 22 = {d3_1+d3_2}")
        return
    
    xmin = d1_1 if d1_1<11 else 11
    ymin = d2_1 if d2_1<13 else 13
    zmin = d3_1 if d3_1<11 else 11
    
    for mri_path,dest_path in entries_sub:
        cmd = f"fslroi {mri_path} {dest_path} {xmin} 160 {ymin} 192 {zmin} 160"
        run(cmd.split(' '), capture_output=True)
        
# fin configuration
t_start = time()
pool = Pool(N_PROCS)

def raise_(e): raise e
for args in zip(ref_paths,list_entries): pool.apply_async(process_entry, args=args, error_callback=raise_)
    
pool.close()
pool.join()