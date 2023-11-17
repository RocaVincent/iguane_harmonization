import nibabel as nib
import pandas as pd
import numpy as np

from tensorflow import convert_to_tensor
from tensorflow.config import list_physical_devices
from model_architecture import Model


#########INPUTS#########
df = pd.read_csv('/NAS/coolio/vincent/data/ixi/metadata_160.csv')
mri_paths = df.sub_id.apply(lambda id_: f"/NAS/coolio/vincent/data/ixi/mni152/n4brains160_normMed/{id_}.nii.gz")
ids_dict = df[['sub_id']].to_dict('list')
def intensity_norm(mri): return mri/500 - 1
def activation(output): return output
model_weights = '/home/vincent/gans_depo/mri/supervised_pred/multicenter_ref/age_pred/model_base_11_160/model.h5'
csv_dest = '/home/vincent/tmp/pred_age.csv'
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
