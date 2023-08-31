import tensorflow as tf


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, is_last, residual_channels, skip_channels, dilation, **kwargs):
        super().__init__(**kwargs)

        self.is_last = is_last
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation

        self.c1 = tf.keras.layers.Conv1D(filters=2 * self.residual_channels, kernel_size=2, padding='causal',
                                         dilation_rate=self.dilation)

        self.c_for_residual = tf.keras.layers.Conv1D(filters=self.residual_channels, kernel_size=1)
        self.c_for_skip = tf.keras.layers.Conv1D(filters=self.skip_channels, kernel_size=1)

        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        z = self.c1(inputs)

        tanh_part = tf.keras.activations.tanh(z[..., :self.residual_channels])
        gate_part = tf.keras.activations.sigmoid(z[..., self.residual_channels:])

        z = tanh_part * gate_part

        if self.is_last == 0:
            for_residual = self.c_for_residual(z)  # if not the last layer, define the residual
            skip = self.c_for_skip(z)

            residual = self.add([for_residual, inputs])
            return residual, skip

        else:
            skip = self.c_for_skip(z)  # only define skip for last layer
            return skip
