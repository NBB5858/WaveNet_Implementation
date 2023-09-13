import json
import tensorflow as tf
from scipy.io.wavfile import write
from utility import *
from WaveNet import WaveNet

# grab parameters to define wavenet #

top_dilation_exp, n_blocks, residual_channels, skips_channels = load_params()

field = (2 ** (top_dilation_exp + 1) - 1) * n_blocks + 1


# create data dictionaries #
train_dict = create_data_dictionary('Training_Data')
val_dict = create_data_dictionary('Validation_Data')

X_val, Y_val = create_features_labels(val_dict)

# load model #
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    name='sparse_categorical_crossentropy'
)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

model = WaveNet(residual_channels=residual_channels, skip_channels=skips_channels,
                field=field, top_dilation_exp=top_dilation_exp, n_blocks=n_blocks)

model.compile(loss=loss_fn, optimizer=optimizer, metrics=['sparse_categorical_crossentropy'])

model(X_val[0][:, :1])

model.load_weights(os.path.join('Saved_Models', 'Best_Model', 'best_model')).expect_partial()

# generate samples #

num_generate = 10
val_file = 0
start, end = 80000, 130000
seed_window = slice(start, end, 1)


seed = X_val[val_file][:, seed_window]

generate_output = model.generate(seed, num_generate)

uncompressed_song = mu_uncompress(generate_output[0, -num_generate:] - 128).astype(np.int16)

write(os.path.join('Generated_Samples', 'song.wav'), 8000, uncompressed_song)
