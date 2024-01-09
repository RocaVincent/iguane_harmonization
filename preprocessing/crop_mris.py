import numpy as np
from nibabel import load as load_mri
from multiprocessing import Pool
from subprocess import run



inp_paths = None # to define
out_paths = None # to define
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

def process_entry(inp_path, out_path):
    print(f"Process IRM {inp_path}")
    d1_1,d1_2,d2_1,d2_2,d3_1,d3_2 = indicesCrop(inp_path)
    if d1_1+d1_2<22:
        print(f"PROBLEM with d1 of {inp_path}, nzeros < 22 = {d1_1+d1_2}")
        return
    if d2_1+d2_2<26:
        print(f"PROBLEM with d2 of {inp_path}, nzeros < 26 = {d2_1+d2_2}")
        return
    if d3_1+d3_2<22:
        print(f"PROBLEM with d3 of {inp_path}, nzeros < 22 = {d3_1+d3_2}")
        return
    
    if d1_1<11: xmin = d1_1
    elif d1_2<11: xmin = 182-160-d1_2
    else: xmin = 11
    
    if d2_1<13: ymin = d2_1
    elif d2_2<13: ymin = 218-192-d2_2
    else: ymin = 13
    
    if d3_1<11: zmin = d3_1
    elif d3_2<11: zmin = 182-160-d3_2
    else: zmin = 11
    
    cmd = f"fslroi {inp_path} {out_path} {xmin} 160 {ymin} 192 {zmin} 160"
    run(cmd.split(' '), capture_output=True)
        
# fin configuration
pool = Pool(N_PROCS)

def raise_(e): raise e
for args in zip(inp_paths,out_paths): pool.apply_async(process_entry, args=args, error_callback=raise_)
    
pool.close()
pool.join()