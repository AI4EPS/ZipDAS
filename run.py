from bls2017_das import decompress
from glob import glob
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="compress", help="compress or decompress")
    parser.add_argument("--model_path", type=str, default="model.h5", help="model path")
    parser.add_argument("--data_path", type=str, default="data", help="data path")
    parser.add_argument("--result_path", type=str, default="compressed", help="result path")
    parser.add_argument("--format", type=str, default="h5", help="data format")
    parser.add_argument("--nt", type=int, default=10240, help="number of time samples")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--plot_figure", action="store_true", help="plot figure")
    parser.add_argument("--workers", type=int, default=1, help="number of threads for preprocessing")
    args = parser.parse_args()
    return args


def read_hdf5(filename):
    """Loads a hdf5 file."""

    if isinstance(filename, tf.Tensor):
        filename = filename.numpy().decode("utf-8")

    with h5py.File(filename, "r") as f:

        data = f["Data"][:]  # (nc, nt)

        data = np.gradient(data, axis=-1)
        data = torch.from_numpy(data).float()

        data = data - torch.median(data, dim=0, keepdim=True)[0]
        data = data - torch.median(data, dim=-1, keepdim=True)[0]

        filter_height, filter_width = 256, 256
        stride_height, stride_width = 256, 256
        in_height, in_width = data.shape
        out_height = math.ceil(in_height / stride_height)
        out_width = math.ceil(in_width / stride_width)

        if in_height % stride_height == 0:
            pad_along_height = max(filter_height - stride_height, 0)
        else:
            pad_along_height = max(filter_height - (in_height % stride_height), 0)
        if in_width % stride_width == 0:
            pad_along_width = max(filter_width - stride_width, 0)
        else:
            pad_along_width = max(filter_width - (in_width % stride_width), 0)

        data = data.unsqueeze(0).unsqueeze(0)  # nb, nc, h, w
        data = F.pad(data, (0, pad_along_width, 0, pad_along_height), mode="reflect")
        avg = F.avg_pool2d(torch.abs(data), kernel_size=256, stride=256)
        # avg = F.avg_pool2d(data**2, kernel_size=256, stride=256)
        # avg = torch.sqrt(avg)
        avg = F.upsample(avg, scale_factor=256, align_corners=False, mode="bilinear")
        data = data / avg

        # return data[0, :, 0:in_height, 0:in_width]
        return data[0, ...]


def load_data(args):

    files = sorted(list(glob(args.data_path + "/*." + args.format)))
    for filename in files:
        data = read_hdf5(filename)
        yield filename, data

    # dataset = tf.data.Dataset.from_tensor_slices(files)
    # dataset = dataset.map(lambda x: tf.py_function(read_hdf5, [x], [tf.string, tf.float32]), num_parallel_calls=args.workers)
    # dataset = dataset.batch(args.batch, drop_remainder=False)


def write_data(args, filename, data):

    with h5py.File(os.path.join(args.result_path, filename.split("/")[-1] + ".h5"), "w") as f:
        f.create_dataset("Data", data=data.numpy())


def plot_result(args, filename, data):

    plt.clf()
    plt.imshow(data, vmin=-1, vmax=1, cmap="seismic")
    plt.savefig(filename, dpi=300)


def compress(args):

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.plot_figure:
        figure_path = os.path.join(args.result_path, "figure")
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

    model = tf.keras.models.load_model(args.model_path)
    dtypes = [t.dtype for t in model.decompress.input_signature]

    dataset = load_data(args)

    nt = args.nt

    for filename, data in dataset:

        vectors = []
        x_shape = []
        y_shape = []

        data = tf.convert_to_tensor(data, dtype=tf.float32)
        nc, h, w = data.shape
        for i in tqdm(range(0, w, nt), desc=f"Compressing {filename.split('/')[-1]}"):
            data_slice = data[:, :, i : i + nt]

            if args.plot_figure:
                plot_result(
                    args,
                    os.path.join(figure_path, filename.split("/")[-1] + f"_{i//nt:02d}.png"),
                    np.transpose(data_slice, (1, 2, 0)),
                )

            data_slice = tf.transpose(data_slice, perm=(1, 2, 0))
            tensors = model.compress(data_slice)
            vectors.append(tensors[0])
            x_shape.append(tensors[1])
            y_shape.append(tensors[2])

        vectors = tf.concat(vectors, axis=0)
        x_shape = tf.concat(x_shape, axis=0)
        y_shape = tf.concat(y_shape, axis=0)

        packed = tfc.PackedTensors()
        packed.pack((vectors, x_shape, y_shape))
        with open(os.path.join(args.result_path, filename.split("/")[-1] + ".tfci"), "wb") as f:
            f.write(packed.string)


def decompress(args):

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.plot_figure:
        figure_path = os.path.join(args.result_path, "figure")
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

    model = tf.keras.models.load_model(args.model_path)
    dtypes = [t.dtype for t in model.decompress.input_signature]

    files = sorted(list(glob(args.data_path + "/*.tfci")))

    for filename in files:

        with open(filename, "rb") as f:
            packed = tfc.PackedTensors(f.read())

        tensors = packed.unpack(dtypes)

        data = []
        for i in tqdm(range(len(tensors[0])), desc=f"Decompressing {filename.split('/')[-1]}"):
            x_hat = model.decompress(
                tensors[0][i : i + 1], tensors[1][i * 2 : i * 2 + 2], tensors[2][i * 2 : i * 2 + 2]
            )

            if args.plot_figure:
                plot_result(args, os.path.join(figure_path, filename.split("/")[-1] + f"_{i:02d}.png"), x_hat)

            data.append(x_hat)

        data = tf.concat(data, axis=1)[:, :, 0]
        write_data(args, filename, data)

    return 0


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "compress":
        compress(args)
    else:
        decompress(args)
