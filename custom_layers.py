# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.legacy import interfaces


class GlobalStandardPooling2D(Layer):
    @interfaces.legacy_global_pooling_support
    def __init__(self, data_format=None, **kwargs):
        super(GlobalStandardPooling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.std(inputs, axis=[1, 2])
        else:
            return K.std(inputs, axis=[2, 3])

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(GlobalStandardPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
