from utility import *
import tensorflow as tf
from WaveNet import WaveNet

# grab parameters to define wavenet #
top_dilation_exp, n_blocks, residual_channels, skips_channels = load_params()
field = (2 ** (top_dilation_exp + 1) - 1) * n_blocks + 1

# define training parameters #
mini_batch_size = int(field * 2)
n_epochs = 1

load_from_weights = False


# define training step functions
@tf.function
def train_step(X_train_mini_batch, Y_train_mini_batch, Wave, loss_fn,
               optimizer, train_mse_metric, train_sparse_cat_cross_metric):

    with tf.GradientTape() as tape:
        Y_train_pred = Wave(X_train_mini_batch, training=True)
        loss = loss_fn(Y_train_mini_batch[:, field-1:, :], Y_train_pred[:, field-1:, :])

    gradients = tape.gradient(loss, Wave.trainable_variables)
    optimizer.apply_gradients(zip(gradients, Wave.trainable_variables))

    train_mse_metric.update_state(Y_train_mini_batch[:, field-1:, 0], tf.argmax(Y_train_pred[:, field-1:, :], axis=2))
    train_sparse_cat_cross_metric.update_state(Y_train_mini_batch[:, field-1:, :], Y_train_pred[:, field-1:, :])

    return


@tf.function
def val_step(X_val_mini_batch, Y_val_mini_batch, Wave, val_mse_metric, val_sparse_cat_cross_metric):

    Y_val_pred = Wave(X_val_mini_batch, training=False)

    val_mse_metric.update_state(Y_val_mini_batch[:, field-1:, 0], tf.argmax(Y_val_pred[:, field-1:, :], axis=2))
    val_sparse_cat_cross_metric.update_state(Y_val_mini_batch[:, field-1:, :], Y_val_pred[:, field-1:, :])

    return


def main():

    train_dict = create_data_dictionary('Training_Data')
    val_dict = create_data_dictionary('Validation_Data')

    X_train, Y_train = create_features_labels(train_dict)
    X_val, Y_val = create_features_labels(val_dict)

    print('training/validation set created')

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        name='sparse_categorical_crossentropy'
    )

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    Wave = WaveNet(residual_channels=residual_channels, skip_channels=skips_channels,
                   field=field, top_dilation_exp=top_dilation_exp, n_blocks=n_blocks)

    Wave.compile(loss=loss_fn,
                 optimizer=optimizer,
                 metrics=['sparse_categorical_crossentropy'])

    if load_from_weights:

        Wave(X_train[0][:, :1])

        Wave.load_weights(os.path.join('Saved_Models', 'Last_Model', 'last_model'))

        train_mse_record, val_mse_record, train_sparse_cat_cross_record, val_sparse_cat_cross_record = load_metrics()

        best_val_loss = min(val_sparse_cat_cross_record)

    else:
        train_mse_record = []
        train_sparse_cat_cross_record = []

        val_mse_record = []
        val_sparse_cat_cross_record = []

        best_val_loss = 1000

    train_mse_metric = tf.keras.metrics.MeanSquaredError()
    train_sparse_cat_cross_metric = tf.keras.metrics.SparseCategoricalCrossentropy()

    val_mse_metric = tf.keras.metrics.MeanSquaredError()
    val_sparse_cat_cross_metric = tf.keras.metrics.SparseCategoricalCrossentropy()

    # Begin training loop
    print('begin training')

    num_train_samples = 0
    for key, song in X_train.items():
        num_train_samples += song.shape[1]

    num_val_samples = 0
    for key, song in X_val.items():
        num_val_samples += song.shape[1]

    num_train_batches = int(num_train_samples / mini_batch_size)
    num_val_batches = int(num_val_samples / mini_batch_size)

    for epoch in range(1, n_epochs + 1):

        print(f"Epoch {epoch}/{n_epochs}")

        # Perform training for each batch
        for m in range(num_train_batches):

            # first, pick a song to look at
            train_song_index = np.random.randint(0, len(X_train))
            X_train_song = X_train[train_song_index]
            Y_train_song = Y_train[train_song_index]

            song_samples = X_train_song.shape[1]

            song_train_start_point = np.random.randint(0, song_samples - mini_batch_size + 2)

            X_train_mini_batch = X_train_song[:, song_train_start_point: song_train_start_point + mini_batch_size]
            Y_train_mini_batch = Y_train_song[:, song_train_start_point: song_train_start_point + mini_batch_size, :]

            train_step(X_train_mini_batch, Y_train_mini_batch, Wave, loss_fn, optimizer,
                       train_mse_metric, train_sparse_cat_cross_metric)

            print(f"\r{m+1}/{num_train_batches} complete", end="" if m+1 < num_train_batches else "\n")

        # evaluate on the validation set

        for m in range(num_val_batches):

            val_song_index = np.random.randint(0, len(X_val))

            X_val_song = X_val[val_song_index]
            Y_val_song = Y_val[val_song_index]

            song_samples = X_val_song.shape[1]

            song_val_start_point = np.random.randint(0, song_samples - mini_batch_size + 2)

            X_val_mini_batch = X_val_song[:, song_val_start_point: song_val_start_point + mini_batch_size]
            Y_val_mini_batch = Y_val_song[:, song_val_start_point: song_val_start_point + mini_batch_size, :]

            val_step(X_val_mini_batch, Y_val_mini_batch, Wave, val_mse_metric, val_sparse_cat_cross_metric)

        print(f'Training root MSE: {tf.sqrt(train_mse_metric.result()):.4f}. Training sparse categorical crossentropy: {train_sparse_cat_cross_metric.result():.4f}')
        print(f'Validation root MSE: {tf.sqrt(val_mse_metric.result()):.4f}. Validation sparse categorical crossentropy: {val_sparse_cat_cross_metric.result():.4f}')

        train_mse_record.append(tf.sqrt(train_mse_metric.result()).numpy())
        train_sparse_cat_cross_record.append(train_sparse_cat_cross_metric.result().numpy())

        val_mse_record.append(tf.sqrt(val_mse_metric.result()).numpy())
        val_sparse_cat_cross_record.append(val_sparse_cat_cross_metric.result().numpy())

        # Save the last model
        Wave.save_weights(os.path.join('Saved_Models', 'Last_Model', 'last_model'), save_format='tf')

        # Save the best model if validation loss improved
        if val_sparse_cat_cross_metric.result() < best_val_loss:
            Wave.save_weights(os.path.join('Saved_Models', 'Best_Model', 'best_model'), save_format='tf')
            print(f'best model saved at epoch {epoch}')
            best_val_loss = val_sparse_cat_cross_metric.result()

        # reset training metrics at the end of an epoch
        train_mse_metric.reset_states()
        train_sparse_cat_cross_metric.reset_states()

        # reset validation metrics at the end of an epoch
        val_mse_metric.reset_states()
        val_sparse_cat_cross_metric.reset_states()

        print("\n")

    print('Training Complete')

    write_metrics(train_mse_record, val_mse_record, train_sparse_cat_cross_record, val_sparse_cat_cross_record)


if __name__ == '__main__':
    main()
