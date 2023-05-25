import pyct ## put top bacause of segmentation fault
import os
import pickle
from collections import defaultdict
from pathlib import Path

import h5py

import imageio.v3 as iio
import numpy as np
import PIL
# import pyct
import pywt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import load_data
from .utils import plot_result

PIL.Image.MAX_IMAGE_PIXELS = None


def compress(args):

    dataset = load_data(args)
    keep_ratio = args.keep_ratio
    result_path = Path(args.result_path)
    if args.plot_figure:
        figure_path = Path(args.figure_path)
    if args.save_preprocess:
        preprocess_path = Path(args.preprocess_path)

    for meta in dataset:

        filename = meta["filename"]
        basename = filename.split("/")[-1]

        if not (result_path / basename).exists():
            (result_path / basename).mkdir(parents=True)

        if args.save_preprocess:
            with h5py.File(preprocess_path / basename, "w") as f:
                f.create_dataset("data", data=meta["data"])

        nx, nt = meta["data"].shape
        if "vmax_uint" in meta:
            vmax_uint = meta["vmax_uint"]
        else:
            vmax_uint = 2**15 - 1

        config = defaultdict(dict)
        for i in tqdm(range(0, nt, args.batch_nt), desc=f"Compressing ({args.method}) {basename}"):

            data = meta["data"][:, i : i + args.batch_nt]

            if args.plot_figure:
                plot_result(
                    args,
                    figure_path / f"{basename}_{i:06d}.png",
                    data,
                )

            if args.method == "wavelet":
                config[f"{i:06d}"] = {
                    "n": 4,
                    "w": "db1",
                }

                coeffs = pywt.wavedec2(data, wavelet=config[f"{i:06d}"]["w"], level=config[f"{i:06d}"]["n"])
                array, coeffs = pywt.coeffs_to_array(coeffs)
                config[f"{i:06d}"]["coeffs"] = coeffs

            elif args.method == "curvelet":
                config[f"{i:06d}"] = {
                    "n": data.shape,
                    "nbs": 3,
                    "nba": 1,
                    "ac": False,
                }
                func_fdct2 = pyct.fdct2(
                    config[f"{i:06d}"]["n"],
                    config[f"{i:06d}"]["nbs"],
                    config[f"{i:06d}"]["nba"],
                    config[f"{i:06d}"]["ac"],
                    norm=False,
                    vec=True,
                    cpx=False,
                )
                array = func_fdct2.fwd(data)

            else:
                config[f"{i:06d}"] = {}
                raise ("method must be wavelet or curvelet")

            topk = int(array.size * (1 - keep_ratio))
            array_1d = np.abs(array.reshape(-1))
            threshold = array_1d[np.argpartition(array_1d, topk)[topk]]
            array_filt = array * (np.abs(array) >= threshold)

            vmax = np.max(np.abs(array_filt))
            array_unit = array_filt / vmax * vmax_uint + vmax_uint
            array_unit = array_unit.astype(np.uint16)

            config[f"{i:06d}"]["vmax"] = vmax

            np.savez_compressed(
                result_path / basename / f"{i:06d}.npz",
                data=array_unit,
            )

        with open((str(result_path / basename) + ".pkl"), "wb") as f:
            tmp = {"mean": meta["mean"], "norm": meta["norm"]}
            if "scale_factor" in meta:
                tmp["scale_factor"] = meta["scale_factor"]
            tmp["vmax_uint"] = vmax_uint
            tmp["config"] = config
            pickle.dump(tmp, f)


def decompress(args):

    data_path = Path(args.data_path)
    result_path = Path(args.result_path)
    if args.plot_figure:
        figure_path = Path(args.figure_path)

    for folder in data_path.glob(f"*.{args.data_format}"):

        with open(str(folder) + ".pkl", "rb") as f:
            norm_dict = pickle.load(f)
            config = norm_dict["config"]
            std = norm_dict["norm"]
            mean = norm_dict["mean"]
            vmax_uint = norm_dict["vmax_uint"]
            if args.recover_amplitude:
                scale_factor = norm_dict["scale_factor"]
                mean_interp = F.interpolate(
                    torch.from_numpy(mean), scale_factor=scale_factor, align_corners=False, mode="bilinear"
                ).numpy()
                std_interp = F.interpolate(
                    torch.from_numpy(std), scale_factor=scale_factor, align_corners=False, mode="bilinear"
                ).numpy()
                std_interp[std_interp == 0] = 1
                mean_interp = np.squeeze(mean_interp)
                std_interp = np.squeeze(std_interp)

        data_list = []
        for filename in tqdm(sorted(list(folder.glob("*.npz"))), desc=f"Decompressing ({args.method}) {folder.name}"):

            meta = np.load(filename)
            array = meta["data"].astype(np.float32)

            config_ = config[filename.stem]

            vmax = config_["vmax"]
            array = (array - vmax_uint) / vmax_uint * vmax

            if args.method == "wavelet":
                coeffs = config_["coeffs"]
                coeffs = pywt.array_to_coeffs(array, coeffs, output_format="wavedec2")
                data = pywt.waverec2(coeffs, wavelet=config_["w"])

            elif args.method == "curvelet":
                func_fdct2 = pyct.fdct2(
                    config_["n"], config_["nbs"], config_["nba"], config_["ac"], norm=False, vec=True, cpx=False
                )
                data = func_fdct2.inv(array)

            else:
                raise ("method must be wavelet or curvelet")

            data_list.append(data)

            if args.plot_figure:
                plot_result(args, figure_path / f"{folder.name}_{filename.name}_{args.method}.png", data)

        data_list = np.concatenate(data_list, axis=-1).astype(np.float32)
        if args.recover_amplitude:
            nx_, nt_ = data_list.shape
            data_list = data_list * std_interp[:nx_, :nt_] + mean_interp[:nx_, :nt_]
        with h5py.File(result_path / folder.name, "w") as f:
            f.create_dataset("data", data=data_list)


def compress_jpeg(args):

    dataset = load_data(args)
    quality = args.quality
    result_path = Path(args.result_path)
    if args.plot_figure:
        figure_path = Path(args.figure_path)
    if args.save_preprocess:
        preprocess_path = Path(args.preprocess_path)

    for meta in dataset:

        filename = meta["filename"]
        basename = filename.split("/")[-1]

        if not (result_path / basename).exists():
            (result_path / basename).mkdir(parents=True)

        if args.save_preprocess:
            with h5py.File(preprocess_path / basename, "w") as f:
                f.create_dataset("data", data=meta["data"])

        nx, nt = meta["data"].shape
        if "vmax_abs" in meta:
            vmax_abs = meta["vmax_abs"]
        else:
            vmax_abs = np.max(np.abs(meta["data"]))
        if "vmax_uint" in meta:
            vmax_uint = meta["vmax_uint"]
        else:
            vmax_uint = 2**15 - 1

        for i in tqdm(range(0, nt, args.batch_nt), desc=f"Compressing (JPEG) {basename}"):

            data = meta["data"][:, i : i + args.batch_nt]

            data = (data / vmax_abs) * vmax_uint + vmax_uint
            data_uint = data.astype(np.uint16)

            if args.plot_figure:
                plot_result(
                    args,
                    figure_path / f"{basename}_{i:06d}.png",
                    data_uint,
                )

            iio.imwrite(
                result_path / basename / f"{i:06d}.j2k",
                data_uint,
                plugin="pillow",
                extension=".j2k",
                quality_mode="rates",
                quality_layers=[quality],
                irreversible=True,
            )

        with open((str(result_path / basename) + ".pkl"), "wb") as f:
            tmp = {"mean": meta["mean"], "norm": meta["norm"]}
            if "scale_factor" in meta:
                tmp["scale_factor"] = meta["scale_factor"]
            if "vmax_abs" in meta:
                tmp["vmax_abs"] = meta["vmax_abs"]
            else:
                tmp["vmax_abs"] = vmax_abs
            if "vmax_uint" in meta:
                tmp["vmax_uint"] = meta["vmax_uint"]
            else:
                tmp["vmax_uint"] = vmax_uint
            pickle.dump(tmp, f)


def decompress_jpeg(args):

    data_path = Path(args.data_path)
    result_path = Path(args.result_path)
    if args.plot_figure:
        figure_path = Path(args.figure_path)

    for folder in data_path.glob(f"*.{args.data_format}"):

        with open(str(folder) + ".pkl", "rb") as f:
            norm_dict = pickle.load(f)
            std = norm_dict["norm"]
            mean = norm_dict["mean"]
            vmax_uint = norm_dict["vmax_uint"]
            vmax_abs = norm_dict["vmax_abs"]
            if args.recover_amplitude:
                scale_factor = norm_dict["scale_factor"]
                mean_interp = F.interpolate(
                    torch.from_numpy(mean), scale_factor=scale_factor, align_corners=False, mode="bilinear"
                ).numpy()
                std_interp = F.interpolate(
                    torch.from_numpy(std), scale_factor=scale_factor, align_corners=False, mode="bilinear"
                ).numpy()
                std_interp[std_interp == 0] = 1
                mean_interp = np.squeeze(mean_interp)
                std_interp = np.squeeze(std_interp)

        data_list = []
        for filename in tqdm(sorted(list(folder.glob("*.j2k"))), desc=f"Decompressing (JPEG) {folder.name}"):

            data = iio.imread(filename, plugin="pillow", extension=".j2k")

            data_list.append(data)

            if args.plot_figure:
                plot_result(args, figure_path / f"{folder.name}_{filename.name}_{args.method}.png", data)

        data_list = np.concatenate(data_list, axis=-1).astype(np.float32)
        data_list = (data_list - vmax_uint) / vmax_uint * vmax_abs
        if args.recover_amplitude:
            nx_, nt_ = data_list.shape
            data_list = data_list * std_interp[:nx_, :nt_] + mean_interp[:nx_, :nt_]
        with h5py.File(result_path / folder.name, "w") as f:
            f.create_dataset("data", data=data_list)


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
