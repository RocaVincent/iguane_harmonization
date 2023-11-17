from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ReLU, BatchNormalization, MaxPool3D, Lambda, Flatten, Dense, Activation
from tensorflow import pad as tf_pad

def Model(image_shape=(160,192,160,1)):
    def conv_block(n_filters, padding1='same'):
        return Sequential([
            Conv3D(filters=n_filters, kernel_size=3, strides=1, padding=padding1, use_bias=False),
            ReLU(),
            Conv3D(filters=n_filters, kernel_size=3, strides=1, padding="same", use_bias=False),
            BatchNormalization(momentum=0.7),
            ReLU(),
            MaxPool3D(pool_size=(2,2,2), strides=(2,2,2))
        ])
    return Sequential([
        Lambda(lambda x: tf_pad(x, ((0,0),(1,1),(1,1),(1,1),(0,0)), 'CONSTANT', constant_values=-1), input_shape=image_shape),
        conv_block(8, padding1='valid'),
        conv_block(16),
        conv_block(32),
        conv_block(64),
        conv_block(128),
        Flatten(),
        Dense(units=1, use_bias=True),
        Activation('linear', dtype='float32')
    ])