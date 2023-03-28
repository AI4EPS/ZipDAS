import math
import os
from glob import glob

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class Config:
    method: str = "Neural"
    mode: str = "compress"
    model_path: str = "model.h5"
    data_path: str = "data"
    result_path: str = "compressed"
    format: str = "h5"

    batch_nt: int = 6000
    normalize_nt: int = 1000
    normalize_nx: int = 1
    batch: int = 1
    plot_figure: bool = False
    workers: int = 1
    max_mad = 10

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def read_hdf5(args, filename):
    """Loads a hdf5 file."""

    with h5py.File(filename, "r") as f:

        data = f["Data"][:]  # (nc, nt)
        # data = f["Data"][:, :4096]  # (nc, nx, nt)
        # data = f["Data"][:, :10240]

        data = np.gradient(data, axis=-1)

        data = torch.from_numpy(data).float()

        return {"data": data, "filename": filename}


def remove_outlier(args, meta):

    data = meta["data"]

    with torch.no_grad():
        MAD = torch.median(torch.abs(data - torch.median(data)))
        vmax = args.max_mad * MAD
        vmin = -vmax
        data = data * (data > vmin) * (data < vmax)

    meta["data"] = data
    return meta


def normalize(args, meta):

    data = meta["data"]
    nx = args.normalize_nx
    nt = args.normalize_nt

    with torch.no_grad():

        filter_height, filter_width = nx, nt
        stride_height, stride_width = filter_height, filter_width
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

        # mean = F.avg_pool2d(data, kernel_size=(nx, nt), stride=(nx, nt))
        # mean = F.interpolate(mean, scale_factor=(nx, nt), align_corners=False, mode="bilinear")
        # data -= mean

        # std = F.avg_pool2d(data**2, kernel_size=256, stride=256)
        # std = torch.sqrt(std)
        std = F.avg_pool2d(torch.abs(data), kernel_size=(nx, nt), stride=(nx, nt))
        std_interp = F.interpolate(std, scale_factor=(nx, nt), align_corners=False, mode="bilinear")
        std_interp[std_interp == 0] = 1
        data /= std_interp

        # data = torch.sign(data) * torch.log(torch.abs(data) + 1.0)
    meta["data"] = data[0, 0, 0:in_height, 0:in_width]
    meta["norm"] = std
    meta["norm_scale_factor"] = [nx, nt]
    return meta


def load_data(args):

    args = Config(**args.__dict__)

    files = sorted(list(glob(args.data_path + "/*." + args.format)))
    print(f"Found {len(files)} files in {args.data_path + '/*.' + args.format}.")

    for filename in files:

        meta = read_hdf5(args, filename)
        meta = remove_outlier(args, meta)
        meta = normalize(args, meta)

        for i in range(0, meta["data"].shape[-1], args.batch_nt):
            yield {"filename": meta["filename"] + f"_{i:06d}", "data": meta["data"][..., i : i + args.batch_nt]}

    # dataset = tf.data.Dataset.from_tensor_slices(files)
    # dataset = dataset.map(lambda x: tf.py_function(read_hdf5, [x], [tf.string, tf.float32]), num_parallel_calls=args.workers)
    # dataset = dataset.batch(args.batch, drop_remainder=False)


def write_data(args, filename, data):

    with h5py.File(os.path.join(args.result_path, filename.split("/")[-1] + ".h5"), "w") as f:
        f.create_dataset("Data", data=data.numpy())
