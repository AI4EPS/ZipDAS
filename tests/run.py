from bls2017_das import decompress
from glob import glob
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.h5', help='model path')
    parser.add_argument('--data_path', type=str, default='data', help='data path')
    parser.add_argument('--result_path', type=str, default='result', help='result path')
    parser.add_argument('--format', type=str, default='h5', help='data format')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--plot_figure', action='store_true', help='plot figure')
    parser.add_argument('--workers', type=int, default=1, help='number of threads for preprocessing')
    args = parser.parse_args()
    return args


def read_hdf5(filename):
    """Loads a hdf5 file."""
    
    if isinstance(filename, tf.Tensor):
        filename_str = filename.numpy().decode('utf-8')

    with h5py.File(filename_str, 'r') as f:

        data = f["Data"][:] # (nc, nt)
        data = np.gradient(data, axis=-1)

        # data = data[:256, :256]
        data = data[256:512, 256:512*2]

        data = (data - np.mean(data, axis=-1, keepdims=True))
        std = np.std(data, axis=-1, keepdims=True)

        print(f'{filename}', np.max(std), np.min(std))

        std[std < 0.01] = 1
        data = data/std
        data = np.sign(data) * np.log(np.abs(data) + 1)

        # return {"data": data[:, :, np.newaxis], "filename": filename}
        # return data[:, :, np.newaxis], filename
        # return filename, data[:1024, :1024, np.newaxis]
        return filename, data[:, :, np.newaxis]
    

def load_data(args):

    files = sorted(list(glob(args.data_path + '/*.' + args.format)))

    dataset = tf.data.Dataset.from_tensor_slices(files)
    # dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x: tf.py_function(read_hdf5, [x], [tf.string, tf.float32]), num_parallel_calls=args.workers)
    # dataset = dataset.batch(args.batch, drop_remainder=False)

    return dataset

def write_compression(args, filename, data):

    if isinstance(filename, tf.Tensor):
        filename_str = filename.numpy().decode('utf-8')
    
    # for f in filename_str:
    name = os.path.basename(filename_str)
    print(name, os.path.join(args.result_path, name))
    with open(os.path.join(args.result_path, name), 'wb') as f:
        f.write(data)


def write_decompression(args, filename, data):
    return data


def plot_result(args, filename, data, raw=None):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(data, vmin=-1, vmax=1, cmap='seismic')
    plt.title('Recovered')
    plt.subplot(1, 2, 2)
    plt.imshow(raw, vmin=-1, vmax=1, cmap='seismic')
    # plt.savefig(os.path.join(args.result_path, filename))
    plt.title("Raw")
    plt.savefig("test.png", dpi=300)

def main(args):

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.plot_figure:
        figure_path = os.path.join(args.result_path, 'figure')
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

    model = tf.keras.models.load_model(args.model_path)

    dataset = load_data(args)

    for x in dataset:

        filename, data = x
        tensors = model.compress(data)

        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors)

        # write_data(args, filename, packed.string)

        x_hat = model.decompress(*tensors)
        print(x_hat.shape)

        if args.plot_figure:
            plot_result(args, filename, x_hat, data)
        
        raise

        # with open(args.output_file, "wb") as f:
        #     f.write(packed.string)

        # # If requested, decompress the image and measure performance.
        # if args.verbose:
        #     x_hat = model.decompress(*tensors)
        #     write_png(args.input_file+"_verbose.png", x_hat)

        #     # Cast to float in order to compute metrics.
        #     x = tf.cast(x, tf.float32)
        #     x_hat = tf.cast(x_hat, tf.float32)
        #     mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        #     psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
        #     msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
        #     msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

        #     # The actual bits per pixel including entropy coding overhead.
        #     num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
        #     bpp = len(packed.string) * 8 / num_pixels


if __name__ == '__main__':
    args = parse_args()
    main(args)