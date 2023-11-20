from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

import json
import sys
sys.path.append('..')
from model_architecture import Model

from input_pipeline.tf_dataset import dataset_from_pathList
from input_pipeline.constants import IMAGE_SHAPE


##########INPUT DATA################
dataset = # to define


#######INPUT PARAMETERS#############
loss = 'mae'
DEST_DIR_PATH = # to define
N_EPOCHS = 400


# optimizer instanciation
INIT_LR = 0.001
END_LR = 0.0001
STEPS_PER_EPOCH = len(mri_paths) // BATCH_SIZE
optimizer = Adam(learning_rate=PolynomialDecay(INIT_LR, N_EPOCHS*STEPS_PER_EPOCH, END_LR))


# model instantiation and compilation
model = Model(image_shape=IMAGE_SHAPE)
model.compile(optimizer=optimizer, loss=loss, jit_compile=True)


# training and saving
history = model.fit(dataset, epochs=N_EPOCHS)

with open(DEST_DIR_PATH+'/stats.json', 'w') as f:
    json.dump(history.history, f)
    
model.save_weights(DEST_DIR_PATH+'/last_model.h5')

