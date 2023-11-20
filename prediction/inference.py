import nibabel as nib
import pandas as pd
import numpy as np

from tensorflow import convert_to_tensor
from tensorflow.config import list_physical_devices
from model_architecture import Model


#########INPUTS#########
mri_paths = # to define
ids_dict = # to define
def intensity_norm(mri): # to define
def activation(output): # to define
model_weights = # to define
csv_dest = # to define
IMAGE_SHAPE = 160,192,160,1
########################

if len(list_physical_devices('GPU'))>0:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")


model = Model(IMAGE_SHAPE)
model.load_weights(model_weights)
ids_dict['prediction'] = np.empty(len(mri_paths))

for i,path in enumerate(mri_paths):
    print(f"Processing image {i+1}/{len(mri_paths)}", end='\r')
    mri = intensity_norm(nib.load(path).get_fdata())
    mri = np.expand_dims(mri, axis=(0,4))
    mri = convert_to_tensor(mri, dtype='float32')
    ids_dict['prediction'][i] = activation(model(mri, training=False).numpy().squeeze())
    
pd.DataFrame(ids_dict).to_csv(csv_dest, index=False)
