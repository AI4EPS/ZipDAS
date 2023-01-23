import argparse
import math
import os
from glob import glob
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import torch
import torch.nn.functional as F
from tqdm import tqdm

from compdas.data import *


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


def plot_result(args, filename, data):

    plt.clf()
    plt.imshow(data, vmin=-1, vmax=1, cmap="seismic")
    plt.savefig(filename, dpi=300)


def download_and_unzip(url, extract_to="./model"):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


def main(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.plot_figure:
        figure_path = os.path.join(args.result_path, "figure")
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        args.figure_path = figure_path

    if not os.path.exists(args.model_path):
        model_url = "https://github.com/AI4EPS/models/releases/download/CompDAS-v1/model.zip"
        download_and_unzip(args.model_path)
        raise

    model = tf.keras.models.load_model(args.model_path)
    dtypes = [t.dtype for t in model.decompress.input_signature]

    if args.mode == "compress":
        compress(args, model, dtypes)

    elif args.mode == "decompress":
        decompress(args, model, dtypes)

    else:
        raise ValueError("Unknown mode")


def compress(args, model, dtypes):

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
                    os.path.join(args.figure_path, filename.split("/")[-1] + f"_{i//nt:02d}.png"),
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


def decompress(args, model, dtypes):

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
                plot_result(args, os.path.join(args.figure_path, filename.split("/")[-1] + f"_{i:02d}.png"), x_hat)

            data.append(x_hat)

        data = tf.concat(data, axis=1)[:, :, 0]
        write_data(args, filename, data)

    return 0


if __name__ == "__main__":
    args = parse_args()
    main(args)
