import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read


def load_params():
    f_params = open('params.json')
    params = json.load(f_params)

    top_dilation_exp = params['top_dilation_exp']
    n_blocks = params['n_blocks']
    residual_channels = params['residual_channels']
    skips_channels = params['skip_channels']

    return top_dilation_exp, n_blocks, residual_channels, skips_channels



def load_from_file(filename, folder):

    samplerate, data = read(os.path.join(folder, filename))

    duration = len(data)/samplerate
    time = np.arange(0, len(data), 1)/samplerate

    return duration, samplerate, time, data


def create_data_dictionary(folder):

    data_dict = {}

    ii = 0
    for filename in os.listdir(folder):
        duration, samplerate, time, data = load_from_file(filename, folder)

        data_dict[ii] = data[8000*10:-8000*10]

        ii += 1

    return data_dict


def mu_compress(data):
    # This maps [-32768, 32767] to [-128,127]

    mu = 255

    scaled_data = data/32768
    eight_bit_data = 128*np.sign(scaled_data)*np.log(1 + mu*np.abs(scaled_data))/np.log(1 + mu)
    eight_bit_data = np.floor(eight_bit_data).astype(int)

    return eight_bit_data


def mu_uncompress(eight_bit_data):
    # This maps [-128, 127] to [-32768  31374]

    mu = 255

    eight_bit_scaled_data = eight_bit_data/128
    sixteen_bit_data = 32768*np.sign(eight_bit_scaled_data)*((mu + 1)**np.abs(eight_bit_scaled_data)-1)/mu
    sixteen_bit_data = np.ceil(sixteen_bit_data).astype(int)

    return sixteen_bit_data


def create_features_labels(data_dict):

    X = {}
    Y = {}

    for key, song in data_dict.items():
        X[key] = mu_compress(song[np.newaxis, :-1].astype(np.float32)) + 128
        Y[key] = mu_compress(song[np.newaxis, 1:, np.newaxis]).astype(np.float32) + 128

    return X, Y


def load_metrics():

    train_mse_record = []
    train_sparse_cat_cross_record = []

    val_mse_record = []
    val_sparse_cat_cross_record = []

    with open(os.path.join('Metrics', 'metrics.csv'), newline='') as f_metrics:

        lines = csv.reader(f_metrics, delimiter=',')
        for index, line in enumerate(lines):
            if index == 0:
                continue
            else:
                train_mse_record.append(float(line[0]))
                val_mse_record.append(float(line[1]))
                train_sparse_cat_cross_record.append(float(line[2]))
                val_sparse_cat_cross_record.append(float(line[3]))

    return train_mse_record, val_mse_record, train_sparse_cat_cross_record, val_sparse_cat_cross_record


def write_metrics(train_mse_record, val_mse_record, train_sparse_cat_cross_record, val_sparse_cat_cross_record):
    with open(os.path.join('Metrics', 'metrics.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['train MSE', 'val MSE', 'train crossentropy', 'val crossentropy'])

        for train_MSE, val_MSE, train_sparse, val_sparse in zip(train_mse_record, val_mse_record,
                                                                train_sparse_cat_cross_record,
                                                                val_sparse_cat_cross_record):
            writer.writerow([train_MSE, val_MSE, train_sparse, val_sparse])


def plot_metrics(train_mse_record, val_mse_record, train_sparse_cat_cross_record, val_sparse_cat_cross_record):

    epochs = range(1, len(train_mse_record) + 1)

    fig, (a1, a2) = plt.subplots(1, 2)

    fig.set_figheight(5)
    fig.set_figwidth(15)

    a1.plot(epochs, train_mse_record, 'r', label='Training')
    a1.plot(epochs, val_mse_record, 'b', label='Validation')

    a1.legend()
    a1.set_title('Root MSE')

    a2.plot(epochs, train_sparse_cat_cross_record, 'r', label='Training')
    a2.plot(epochs, val_sparse_cat_cross_record, 'b', label='Validation')

    a2.legend()
    a2.set_title('Categorical Crossentropy')

    fig.show()

    return
