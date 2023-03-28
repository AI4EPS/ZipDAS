### segmentation fault
import pyct
import pywt

###
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
from collections import defaultdict
import re
from compdas.data import *
import json
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="Neural", help="Neural, Wavelet, Curvelet")
    parser.add_argument("--mode", type=str, default="compress", help="compress or decompress")
    parser.add_argument("--model_path", type=str, default="model.h5", help="model path")
    parser.add_argument("--data_path", type=str, default="data", help="data path")
    parser.add_argument("--result_path", type=str, default="compressed", help="result path")
    parser.add_argument("--format", type=str, default="h5", help="data format")
    parser.add_argument("--keep_ratio", type=float, default=0.1, help="keep ratio")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    # parser.add_argument("--batch_window", type=int, default=10240, help="number of time samples")
    parser.add_argument("--plot_figure", action="store_true", help="plot figure")
    parser.add_argument("--workers", type=int, default=1, help="number of threads for preprocessing")
    args = parser.parse_args()
    return args


def plot_result(args, filename, data):

    plt.clf()
    # plt.imshow(data, vmin=-1, vmax=1, cmap="seismic")
    vmax = np.std(data) * 3
    vmin = -vmax
    plt.matshow(data, vmin=vmin, vmax=vmax, cmap="seismic")
    plt.savefig(filename, dpi=300)
    plt.close()


def download_and_unzip(url, extract_to="./model"):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


def main(args):

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.result_path = os.path.join(args.result_path, args.method)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if args.plot_figure:
        args.figure_path = os.path.join(args.result_path, "figures")
        if not os.path.exists(args.figure_path):
            os.makedirs(args.figure_path)

    if args.mode == "compress":

        if not os.path.exists(args.model_path):
            model_url = "https://github.com/AI4EPS/models/releases/download/CompDAS-v2/model.zip"
            download_and_unzip(args.model_path)
            raise

        if args.method == "Neual":
            model = tf.keras.models.load_model(args.model_path)
            dtypes = [t.dtype for t in model.decompress.input_signature]
            compress_neural(args, model, dtypes)

        elif (args.method == "wavelet") or (args.method == "curvelet"):
            compress(args)

        else:
            raise ("method must be neural, wavelet, or curvelet")

    elif args.mode == "decompress":

        if args.method == "Neual":
            model = tf.keras.models.load_model(args.model_path)
            dtypes = [t.dtype for t in model.decompress.input_signature]
            decompress_neural(args, model, dtypes)

        elif (args.method == "wavelet") or (args.method == "curvelet"):
            decompress(args)

        else:
            raise ("method must be neural, wavelet, or curvelet")
    else:
        raise ("mode must be compress or decompress")


def compress(args):

    dataset = load_data(args)

    for meta in tqdm(dataset):

        data = meta["data"].numpy()
        filename = meta["filename"]

        if args.plot_figure:
            plot_result(args, os.path.join(args.figure_path, filename.split("/")[-1] + f"_{args.method}.png"), data)

        nx, nt = data.shape
        keep = args.keep_ratio
        meta = {}

        if args.method == "wavelet":

            config = {
                "n": 4,
                "w": "db1",
            }

            coeffs = pywt.wavedec2(data, wavelet=config["w"], level=config["n"])
            array, coeffs = pywt.coeffs_to_array(coeffs)
            config["coeffs"] = coeffs

        elif args.method == "curvelet":
            nx, nt = data.shape
            config = {
                "n": (nx, nt),
                "nbs": 3,
                "nba": 1,
                "ac": False,
            }
            func_fdct2 = pyct.fdct2(
                config["n"], config["nbs"], config["nba"], config["ac"], norm=False, vec=True, cpx=False
            )
            array = func_fdct2.fwd(data)

        else:
            config = {}
            raise ("method must be wavelet or curvelet")

        topk = int(array.size * (1 - keep))
        array_1d = np.abs(array.reshape(-1))
        threshold = array_1d[np.argpartition(array_1d, topk)[topk]]
        array_filt = array * (np.abs(array) >= threshold)
        # print("None zeros ratios: ", np.count_nonzero(array_filt) / array.size)

        vrange = [array_filt.min(), array_filt.max()]
        array_quant = ((array_filt - vrange[0]) / (vrange[1] - vrange[0]) * 255.0).astype(np.uint8)
        config["vrange"] = vrange

        np.savez_compressed(
            os.path.join(args.result_path, filename.split("/")[-1] + f".npz"),
            allow_pickle=False,
            data=array_quant,
            # vrange=vrange,
            # **config,
        )

        ## save config to pickle file
        with open(os.path.join(args.result_path, filename.split("/")[-1] + f".pkl"), "wb") as f:
            pickle.dump(config, f)


def decompress(args):

    files = sorted(list(glob(args.data_path + "/*.npz")))

    data_list = []
    h5_name = ""
    for filename in tqdm(files):

        new_name = re.sub(r"_[0-9]*.npz$", "", filename.split("/")[-1])
        if new_name != h5_name:
            if len(data_list) > 0:
                with h5py.File(os.path.join(args.result_path, h5_name), "w") as f:
                    f.create_dataset("data", data=np.concatenate(data_list, axis=-1))
            h5_name = new_name
            data_list = []

        meta = np.load(filename, allow_pickle=(args.method == "wavelet"))
        with open(filename.replace(".npz", ".pkl"), "rb") as f:
            config = pickle.load(f)

        array = meta["data"].astype(np.float32)
        vrange = config["vrange"]
        array = array.astype(np.float32) * (vrange[1] - vrange[0]) / 255.0 + vrange[0]

        if args.method == "wavelet":
            # config = {
            #     "n": int(meta["n"]),
            #     "w": meta["w"],
            # }

            coeffs = config["coeffs"]
            coeffs = pywt.array_to_coeffs(array, coeffs, output_format="wavedec2")
            data = pywt.waverec2(coeffs, wavelet=config["w"])

        elif args.method == "curvelet":
            # config = {
            #     "n": tuple([int(x) for x in meta["n"]]),
            #     "nbs": int(meta["nbs"]),
            #     "nba": int(meta["nba"]),
            #     "ac": bool(meta["ac"]),
            # }

            func_fdct2 = pyct.fdct2(
                config["n"], config["nbs"], config["nba"], config["ac"], norm=False, vec=True, cpx=False
            )
            data = func_fdct2.inv(array)

        else:
            raise ("method must be wavelet or curvelet")

        if args.plot_figure:
            plot_result(args, os.path.join(args.figure_path, filename.split("/")[-1] + f"_{args.method}.png"), data)

    if len(data_list) > 0:
        with h5py.File(os.path.join(args.result_path, h5_name), "w") as f:
            f.create_dataset("data", data=np.concatenate(data_list, axis=-1))


def compress_neural(args, model, dtypes):

    dataset = load_data(args)

    nt = args.batch_window

    for meta in dataset:
        data = meta["data"].unsqueeze(0)
        filename = meta["filename"]

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


def decompress_neural(args, model, dtypes):

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
