import math
import os
from glob import glob

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def read_hdf5(args, filename, nx=256, nt=256):
    """Loads a hdf5 file."""

    with h5py.File(filename, "r") as f:

        data = f["Data"][:]  # (nc, nt)

        data = np.gradient(data, axis=-1)
        data = torch.from_numpy(data).float()

        data = data - torch.median(data, dim=0, keepdim=True)[0]
        data = data - torch.median(data, dim=-1, keepdim=True)[0]

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
        avg = F.avg_pool2d(torch.abs(data), kernel_size=(nx, nt), stride=(nx, nt))
        # avg = F.avg_pool2d(data**2, kernel_size=256, stride=256)
        # avg = torch.sqrt(avg)
        avg = F.interpolate(avg, scale_factor=(nx, nt), align_corners=False, mode="bilinear")
        data = data / avg

        data = torch.sign(data) * torch.log(torch.abs(data) + 1.0)

        # return data[0, :, 0:in_height, 0:in_width]
        return data[0, ...]


def load_data(args):

    files = sorted(list(glob(args.data_path + "/*." + args.format)))
    print(f"Found {len(files)} files in {args.data_path + '/*.' + args.format}.")
    for filename in files:
        data = read_hdf5(args, filename)
        yield filename, data

    # dataset = tf.data.Dataset.from_tensor_slices(files)
    # dataset = dataset.map(lambda x: tf.py_function(read_hdf5, [x], [tf.string, tf.float32]), num_parallel_calls=args.workers)
    # dataset = dataset.batch(args.batch, drop_remainder=False)


def write_data(args, filename, data):

    with h5py.File(os.path.join(args.result_path, filename.split("/")[-1] + ".h5"), "w") as f:
        f.create_dataset("Data", data=data.numpy())