import numpy as np
import tensorflow as tf
from ResidualLayer import ResidualLayer


class WaveNet(tf.keras.Model):
    def __init__(self, residual_channels, skip_channels, field, top_dilation_exp, n_blocks, **kwargs):
        super().__init__(**kwargs)

        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.field = field
        self.top_dilation_exp = top_dilation_exp
        self.n_blocks = n_blocks

        is_last = 0

        self.first_layer = tf.keras.layers.Conv1D(filters=self.residual_channels, kernel_size=2,
                                                  padding='causal', activation='tanh')

        self.blocks = dict()
        for bb in range(n_blocks):
            self.blocks[f'block_{bb}'] = []
            for ii in [2**jj for jj in range(self.top_dilation_exp + 1)]:

                if bb == n_blocks - 1 and ii == 2**top_dilation_exp:
                    is_last = 1

                self.blocks[f'block_{bb}'].append(ResidualLayer(is_last, self.residual_channels,
                                                                self.skip_channels, ii))

        self.add = tf.keras.layers.Add()
        self.l1x1relu = tf.keras.layers.Conv1D(filters=256, kernel_size=1, activation='relu')
        self.l1x1 = tf.keras.layers.Conv1D(filters=256, kernel_size=1)

        self.out = tf.keras.layers.Softmax(axis=2)

    def build(self, batch_input_shape):

        self.embedding_layer = tf.keras.layers.Embedding(256, self.residual_channels)
        super().build(batch_input_shape)

    def call(self, inputs):

        z = self.embedding_layer(inputs)

        z = self.first_layer(z)

        skips = []
        for bb in range(self.n_blocks):
            for index, layer in enumerate(self.blocks[f'block_{bb}']):

                if bb == self.n_blocks - 1 and index == self.top_dilation_exp:
                    skip_add = layer(z)
                else:
                    z, skip_add = layer(z)

                skips.append(skip_add)

        z = tf.keras.activations.relu(self.add(skips))
        z = self.l1x1relu(z)
        z = self.l1x1(z)

        return self.out(z)

    def generate(self, inputs, n_samples):

        print('begin inference')

        input_length = inputs.shape[1]

        song = np.zeros((1, input_length + n_samples))
        song[:, :input_length] = inputs

        for ii in range(0, n_samples):

            input_slice = song[:, input_length - self.field + ii:input_length + ii]

            pred = np.array(self.call(input_slice)[0, -1:, :]).argmax()

            song[0, input_length + ii] = pred

            if (ii+1) % 100 == 0:
                print(f'inference {ii+1} complete')

        return song
