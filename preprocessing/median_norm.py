from nibabel import load as load_mri, save as save_mri, Nifti1Image
from multiprocessing import Pool
import numpy as np


mri_paths = None # to define
mask_paths = None # to define
dest_paths = None # to define
N_PROCS = None # to define
MEDIAN = 500


def process_mri(mri_path, mask_path, dest_path):
    print(f"Start processing {mri_path}")
    mri = load_mri(mri_path)
    data = mri.get_fdata()
    mask = load_mri(mask_path).get_fdata()>0
    med = np.median(data[mask])
    data /= (med/MEDIAN)
    mri = Nifti1Image(data, affine=mri.affine, header=mri.header)
    save_mri(mri, dest_path)

pool = Pool(N_PROCS)
def raise_(e): raise e
for args in zip(mri_paths, mask_paths, dest_paths):
    pool.apply_async(process_mri, args=args, error_callback=raise_)

pool.close()
pool.join()