from scipy.ndimage import rotate
from tensorflow.random import uniform
from tensorflow import function as tf_function, roll as tf_roll, pad as tf_pad, constant as tf_constant, numpy_function
from .constants import IMAGE_SHAPE


@tf_function(jit_compile=False) # jit compile problem pad
def random_shift_vec(mri, shift_max, cval):
    """
    mri : 4D (width, height, depth, n_channels) tensor
    Randomly shiftes mri in all dimension (width, height, depth).
    For each dimension, shiftes between 0 and shift_max_voxels
    """
    shifts = uniform((3,), minval=-shift_max, maxval=shift_max+1, dtype="int32")
    mri = tf_roll(mri, shift=shifts, axis=(0,1,2))
    
    # cropping
    cropping = ((shifts[0],0) if shifts[0]>=0 else (0,-shifts[0]),
               (shifts[1],0) if shifts[1]>=0 else (0,-shifts[1]),
               (shifts[2],0) if shifts[2]>=0 else (0,-shifts[2]))
    mri = mri[cropping[0][0]:IMAGE_SHAPE[0]-cropping[0][1],
              cropping[1][0]:IMAGE_SHAPE[1]-cropping[1][1],
              cropping[2][0]:IMAGE_SHAPE[2]-cropping[2][1]:,]
    
    # apply padding
    return tf_pad(mri, paddings=(cropping[0],cropping[1],cropping[2],(0,0)), mode="CONSTANT", constant_values=cval)


axes_list = tf_constant([[0,1],[0,2],[1,2]], dtype='int32')

def rotation(mri, angle, axes, cval):
    return rotate(mri, angle=angle, axes=axes, reshape=False, mode="constant", cval=cval)

@tf_function # jit compile incompatible avec numpy_function
def random_rotation_vec(mri, angle_max, cval):
    """
    mri : 4D (width, height, depth, n_channels)
    Randomly rotates MRIs in a randomly drawn plan (among 3 plans)
    """
    angle = uniform((), minval=-angle_max, maxval=angle_max, dtype="float32")
    axes = axes_list[uniform((), minval=0, maxval=3, dtype="int32")]
    mri = numpy_function(rotation, inp=[mri, angle, axes, cval], Tout='float32')
    mri.set_shape(IMAGE_SHAPE)
    return mri

@tf_function
def augmentation(mri):
    if uniform(shape=(1,), minval=0, maxval=1, dtype="float32") < 0.5:
        mri = random_rotation_vec(mri, angle_max=10, cval=-1)
    return random_shift_vec(mri, shift_max=5, cval=-1)