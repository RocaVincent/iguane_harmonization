from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Layer, Conv3D, Conv3DTranspose, ReLU, Activation, Add, LeakyReLU, Concatenate, Lambda
from tensorflow.math import reduce_mean, abs as tf_abs
from tensorflow import pad as tf_pad

class InstanceNormalization(Layer):
    
    def __init__(self, epsilon=1e-3):
        super().__init__(dtype='float32')
        self.epsilon = epsilon
        
    def build(self, batch_input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=batch_input_shape[-1:],
            initializer="ones",
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=batch_input_shape[-1:],
            initializer="zeros",
            trainable=True
        )
        self.axis = range(1, len(batch_input_shape)-1)
        super().build(batch_input_shape)
        
    def call(self, x):
        mean = reduce_mean(x, axis=self.axis, keepdims=True)
        dev = x-mean
        mean_abs_dev = reduce_mean(tf_abs(dev), axis=self.axis, keepdims=True)
        normalized = dev / (mean_abs_dev+self.epsilon)
        return self.scale*normalized + self.offset

def Generator(image_shape=(None,None,None,1), kernel_initializer='glorot_uniform'):
    # le générateur façon deepHarmony, version résiduelle
    def Conv_block(n_filters):
        return Sequential([
          Conv3D(filters=n_filters, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer),
          ReLU(),
          InstanceNormalization()
        ])

    def Downsample_block(n_filters):
        return Sequential([
          Conv3D(filters=n_filters, kernel_size=4, strides=2, padding='same', kernel_initializer=kernel_initializer),
          ReLU(),
          InstanceNormalization()
        ])

    def Upsample_block(n_filters):
        return Sequential([
          Conv3DTranspose(filters=n_filters, kernel_size=4, strides=2, padding='same', kernel_initializer=kernel_initializer),
          ReLU(),
          InstanceNormalization()
        ])

    inputs = Input(shape=image_shape)
    conv1 = Conv_block(16)(inputs)
    x = Downsample_block(16)(conv1)
    conv2 = Conv_block(32)(x)
    x = Downsample_block(32)(conv2)
    conv3 = Conv_block(64)(x)
    x = Downsample_block(64)(conv3)
    conv4 = Conv_block(128)(x)
    x = Downsample_block(128)(conv4)

    x = Conv_block(256)(x)
    x = Upsample_block(128)(x)
    x = Concatenate()([x,conv4])
    x = Conv_block(128)(x)
    x = Upsample_block(64)(x)
    x = Concatenate()([x,conv3])
    x = Conv_block(64)(x)
    x = Upsample_block(32)(x)
    x = Concatenate()([x,conv2])
    x = Conv_block(32)(x)
    x = Upsample_block(16)(x)
    x = Concatenate()([x,conv1])
    x = Conv_block(16)(x)
    x = Concatenate()([x,inputs])
    last = Sequential([
        Conv3D(filters=1, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_initializer),
        Activation('tanh')
    ])(x)
    return Model(inputs=inputs, outputs=Add(dtype='float32')([last,inputs]))


def Discriminator(image_shape=(None,None,None,1), lrelu_slope=0.2, kernel_initializer='glorot_uniform'):
    
    def Downsample(n_filters, padding='same', inst_norm=True):
        model = Sequential([
            Conv3D(filters=n_filters, kernel_size=4, strides=2, padding=padding, use_bias=False, kernel_initializer=kernel_initializer)
        ])
        if inst_norm:
            model.add(InstanceNormalization())
        model.add(LeakyReLU(lrelu_slope))
        return model
    
    return Sequential([
        Lambda(lambda x: tf_pad(x, ((0,0),(1,1),(1,1),(1,1),(0,0)), 'CONSTANT', constant_values=-1), input_shape=image_shape),
        Downsample(64, padding='valid', inst_norm=False),
        Downsample(128),
        Downsample(256),
        Conv3D(filters=1, kernel_size=3, strides=1, padding="same", use_bias=True, kernel_initializer=kernel_initializer),
        Activation('linear', dtype='float32')
    ])


