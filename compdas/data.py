import math
import os
from dataclasses import dataclass
from glob import glob

import h5py
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Config:

    data_format: str = "h5"
    normalize_nt: int = 6000
    normalize_nx: int = 1
    sensitivity = 6.769989802467751e-03 # micro m/s for Redgecrest

    workers: int = 1
    # percentile = 0.999
    percentile = 0.99

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def read_hdf5(args, filename):
    """Loads a hdf5 file."""

    with h5py.File(filename, "r") as f:

        if "Data" in f:
            data = f["Data"][:]  # (nc, nt)
            data = np.gradient(data, f["Data"].attrs["dt"], axis=-1) * args.sensitivity
        elif "data" in f:
            data = f["data"][:].T
        else:
            raise ValueError(f"{filename} does not contain Data or data")

        data = data[:, :6000]
        return {"data": data, "filename": filename}
    
# def read_memmap(args, filename):

#     """Loads a memmap file."""

#     data = np.memmap(filename, dtype='float32', mode='r', shape=tuple(args.template_shape))
#     data = torch.tensor(data, dtype=torch.float32)

#     return {"data": data, "filename": filename}


def percentile_filter(args, meta):

    data = meta["data"]
    percentile = args.percentile

    # with torch.no_grad():
        
        # MAD = torch.median(torch.abs(data - torch.median(data, dim=(-2,-1), keepdim=True)[0]), dim=(-2,-1), keepdim=True)[0]
        # print(f"{MAD = } {data.max() = } {data.min() = }")
        # vmax = args.max_mad * MAD

        # std = torch.std(data, dim=(-2,-1), keepdim=True)
        # vmax = args.max_mad * std
        # vmin = -vmax

        # vmax = torch.quantile(torch.abs(data), 0.99)
        # vmin = -vmax
        # data = data * (data > vmin) * (data < vmax)

        # https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html
        # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
        # https://en.wikipedia.org/wiki/Histogram_equalization

        # hist = np.histogram(array_quant, bins=256, range=(0, 256))
        # cdf = hist.cumsum()
        # cdf_m = np.ma.masked_equal(cdf,0)
        # cdf_m = (cdf_m - cdf_m.min())*255.0/(cdf_m.max()-cdf_m.min())
        # cdf = np.ma.filled(cdf_m,0).astype(np.uint8)

        # array_quant = cv2.equalizeHist(array_quant.astype(np.uint8))
        # print(array_quant[:10, :10])
        # raise
        # array_quant = cdf[array_quant]

    vmax = np.quantile(np.abs(data), percentile)
    vmin = -vmax

    # from scipy.stats import norm
    # def z_to_percentile(z_score):
    #     return norm.cdf(z_score)
    # z_score = 4.0  # change this to your z_score
    # percentile = z_to_percentile(z_score)
    # print(f"The percentile of Z = {z_score} is {percentile}")

    # vmax = np.std(data) * 6.0
    # vmin = -vmax

    data = np.clip(data, vmin, vmax)
    vmax = np.max(np.abs(data))

    meta["data"] = data
    # meta["vmin"] = vmin
    # meta["vmax"] = vmax
    meta["vmax_abs"] = vmax

    return meta

def equalize_hist(args, meta):
    data = meta["data"]
    vmax = np.max(np.abs(data))
    vmin = -vmax
    vrange = 2**15 - 1
    # data = (data - vmin) / (vmax - vmin) * (2**16-1)
    data = (data / vmax) * vrange + vrange
    data = data.astype('uint16')
    hist, bins = np.histogram(data.flatten(), np.arange(0, 2**16))
    # hist, bins = np.histogram(data.flatten(), 2**16, [0,2**16])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*(2**16 - 1)/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint16')
    data = cdf[data]
    # data = data/(2**16-1) * (vmax - vmin) + vmin
    data = (data.astype('float32') - vrange) / vrange * vmax
    meta["data"] = data

    meta["vmax_abs"] = vmax
    meta["vmax_uint"] = vrange
    return meta

def normalize(args, meta):

    data = meta["data"]

    data = torch.from_numpy(data).float()
    
    nx = args.normalize_nx
    nt = args.normalize_nt

    if data.shape[-1] < nt:
        nx_, nt_ = data.shape
        mean = torch.mean(data, dim=-1, keepdim=True)
        data -= mean
        std = torch.mean(np.abs(data), dim=-1, keepdim=True)
        std[std == 0] = 1
        data /= std

        meta["data"] = data.numpy().astype(np.float32)
        meta["mean"] = mean.numpy().astype(np.float32)
        meta["norm"] = std.numpy().astype(np.float32)
        return meta
    
    num_dim = len(data.shape)

    with torch.no_grad():

        filter_height, filter_width = nx, nt
        stride_height, stride_width = filter_height, filter_width
        in_height, in_width = data.shape[-2:]
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
        
        if num_dim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # nb, nc, h, w
        elif num_dim == 3:
            data = data.unsqueeze(0)
        else:
            pass

        data = F.pad(data, (0, pad_along_width, 0, pad_along_height), mode="reflect")

        mean = F.avg_pool2d(data, kernel_size=(nx, nt), stride=(nx, nt))
        mean_interp = F.interpolate(mean, scale_factor=(nx, nt), align_corners=False, mode="bilinear")
        data -= mean_interp

        # std = F.avg_pool2d(data**2, kernel_size=256, stride=256)
        # std = torch.sqrt(std)
        std = F.avg_pool2d(torch.abs(data), kernel_size=(nx, nt), stride=(nx, nt))
        std_interp = F.interpolate(std, scale_factor=(nx, nt), align_corners=False, mode="bilinear")
        std_interp[std_interp == 0] = 1
        data /= std_interp

        # data = torch.sign(data) * torch.log(torch.abs(data) + 1.0)
    
    if num_dim == 2:
        data = data.squeeze(0).squeeze(0)
    elif num_dim == 3:
        data = data.squeeze(0)
    else:
        pass

    meta["data"] = data[0:in_height, 0:in_width].numpy().astype(np.float32)
    meta["mean"] = mean.numpy().astype(np.float32)
    meta["norm"] = std.numpy().astype(np.float32)
    meta["scale_factor"] = [nx, nt]

    return meta


def load_data(args):

    args = Config(**args.__dict__)

    print(f"\nFound {len(glob(args.data_path + '/*.' + args.data_format))} files in {args.data_path + '/*.' + args.data_format}.")

    if args.data_format == "h5":
        files = sorted(list(glob(args.data_path + "/*." + args.data_format)))
        
    elif args.data_format == "dat":
        mmap_data = np.memmap(glob(args.data_path + "/*." + args.data_format)[0], dtype='float32', mode='r', shape=tuple(args.template_shape))
        files = range(args.template_shape[0])

    for filename in files:

        if args.data_format == "h5":
            meta = read_hdf5(args, filename)
        elif args.data_format == "dat":
            meta = {"data": torch.tensor(mmap_data[filename], dtype=torch.float32), "filename": filename}
        else:
            raise ValueError(f"{args.data_format} is not supported.")
        
        # import matplotlib.pyplot as plt
        # idx0 = 3000
        # idx1 = 9000
        # idx0 = 6000
        # idx1 = 12000
        # # idx0 = 0
        # # idx1 = 6000
        # data = meta["data"][:,idx0:idx1].copy()
        # data_raw = data.copy()
        # type = "noise"
        # # type = "event"
        # print(data.shape, data.min(), data.max())

        # fig, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 4.5), gridspec_kw={"height_ratios": [2, 1]})
        # if type == "event":
        #     vmax = np.std(data_raw) 
        # else:
        #     vmax = np.std(data_raw) / 200
        # vmin = -vmax
        # axes[0, 0].imshow(data, vmin=vmin, vmax=vmax, cmap="seismic")
        # axes[0, 0].set_xlabel("Time index")
        # axes[0, 0].set_ylabel("Channel index")
        # axes[1, 0].hist(data.reshape(-1), bins=100, range=(vmin, vmax))
        # axes[1, 0].set_xlabel("Strain rate (1e-6)")
        # axes[1, 0].set_ylabel("Frequency")
        # axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.savefig(f"debug_raw_{type}.pdf", dpi=300, bbox_inches="tight")


        # data_raw = meta["data"][:,idx0:idx1].copy()
        meta = percentile_filter(args, meta)
        # data = meta["data"][:,idx0:idx1].copy()

        # fig, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 4.5), gridspec_kw={"height_ratios": [1, 1]})
        # if type == "event":
        #     vmax = np.std(data)
        # else:
        #     vmax = np.std(data_raw) / 200
        # vmin = -vmax
        # diff = data - data_raw
        # axes[0, 0].imshow(data_raw, vmin=vmin, vmax=vmax, cmap="seismic")
        # # axes[0, 0].set_xlabel("Time index")
        # axes[0, 0].set_ylabel("Channel index")
        # axes[1, 0].imshow(diff, vmin=vmin, vmax=vmax, cmap="seismic")
        # axes[1, 0].set_xlabel("Time index")
        # axes[1, 0].set_ylabel("Channel index")
        # plt.savefig(f"debug_percentile1_{type}.pdf", dpi=300, bbox_inches="tight")

        # data_raw = meta["data"][:,idx0:idx1].copy()
        meta = normalize(args, meta)
        # data = meta["data"][:,idx0:idx1].copy()

        # fig, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 4.5), gridspec_kw={"height_ratios": [2, 1]})
        # # data -= np.mean(data, axis=-1, keepdims=True)
        # # data /= np.std(data, axis=-1, keepdims=True)
        # vmax = 3
        # vmin = -3
        # axes[0, 0].imshow(data, vmin=vmin, vmax=vmax, cmap="seismic")
        # axes[0, 0].set_xlabel("Time index")
        # axes[0, 0].set_ylabel("Channel index")
        # axes[1, 0].hist(data.reshape(-1), bins=100, range=(vmin, vmax))
        # axes[1, 0].set_xlabel("Normalized amplitude")
        # axes[1, 0].set_ylabel("Frequency")
        # axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # plt.savefig(f"debug_normalized_{type}.pdf", dpi=300, bbox_inches="tight")

        # data_raw = meta["data"][:,idx0:idx1].copy()
        meta = percentile_filter(args, meta)
        # data = meta["data"][:,idx0:idx1].copy()
        
        # fig, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 4.5), gridspec_kw={"height_ratios": [1, 1]})
        # mean = np.mean(data_raw, axis=-1, keepdims=True)
        # std = np.std(data_raw, axis=-1, keepdims=True)
        # diff = data - data_raw
        # data_raw -= mean
        # data_raw /= std
        # data -= mean
        # data /= std
        # diff /= std
        # vmax = 3
        # vmin = -3
        # axes[0, 0].imshow(data_raw, vmin=vmin, vmax=vmax, cmap="seismic")
        # axes[0, 0].set_ylabel("Channel index")
        # axes[1, 0].imshow(diff, vmin=vmin, vmax=vmax, cmap="seismic")
        # axes[1, 0].set_xlabel("Time index")
        # axes[1, 0].set_ylabel("Channel index")
        # plt.savefig(f"debug_percentile2_{type}.pdf", dpi=300, bbox_inches="tight")

        # data_raw = meta["data"][:,idx0:idx1].copy()
        meta = equalize_hist(args, meta)
        # data = meta["data"][:,idx0:idx1].copy().astype(np.float32)

        # fig, axes = plt.subplots(3, 1, squeeze=False, figsize=(10, 8.5), gridspec_kw={"height_ratios": [1, 1, 1]})
        # mean = np.mean(data_raw, axis=-1, keepdims=True)
        # std = np.std(data_raw, axis=-1, keepdims=True)
        
        # data_raw -= mean
        # data_raw /= std
        # mean = np.mean(data, axis=-1, keepdims=True)
        # std = np.std(data, axis=-1, keepdims=True)
        # data -= mean
        # data /= std

        # diff = data - data_raw
        # vmax = 3
        # vmin = -3
        # axes[0, 0].imshow(data_raw, vmin=vmin, vmax=vmax, cmap="seismic")
        # axes[1, 0].set_xlabel("Time index")
        # axes[0, 0].set_ylabel("Channel index")
        # axes[1, 0].imshow(data, vmin=vmin, vmax=vmax, cmap="seismic")
        # axes[1, 0].set_xlabel("Time index")
        # axes[1, 0].set_ylabel("Channel index")
        # axes[2, 0].hist(data_raw.reshape(-1), bins=200, range=(vmin, vmax))
        # axes[2, 0].hist(data.reshape(-1), bins=200, range=(vmin, vmax), alpha=0.8)
        # axes[2, 0].set_xlabel("Normalized amplitude")
        # axes[2, 0].set_ylabel("Frequency")
        # axes[2, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
        # plt.savefig(f"debug_equalize_hist_{type}.pdf", dpi=300, bbox_inches="tight")

        yield meta

        # data = meta["data"]
        # if len(data.shape) == 2:
        #     for k in range(0, data.shape[-1], args.batch_nt):
        #         yield {"filename": f'{meta["filename"]}' + f"_{k:06d}", "data": data[:, k : k + args.batch_nt], "norm": meta["norm"], "mean": meta["mean"], "norm_scale_factor": meta["norm_scale_factor"]}
        # elif len(data.shape) == 3:
        #     for j in range(data.shape[0]):
        #         for k in range(0, data.shape[-1], args.batch_nt):
        #             yield {"filename": f'{meta["filename"]}' + f"_{j:06d}_{k:06d}", "data": data[j, :, k : k + args.batch_nt], "norm": meta["norm"], "mean": meta["mean"], "norm_scale_factor": meta["norm_scale_factor"]}
        # elif len(data.shape) == 4:
        #     for i in range(data.shape[0]):
        #         for j in range(data.shape[1]):
        #             for k in range(0, data.shape[-1], args.batch_nt):
        #                 yield {"filename": f'{meta["filename"]}' + f"_{i:06d}_{j:06d}_{k:06d}", "data": data[i, j, :, k : k + args.batch_nt], "norm": meta["norm"], "mean": meta["mean"], "norm_scale_factor": meta["norm_scale_factor"]}

    # dataset = tf.data.Dataset.from_tensor_slices(files)
    # dataset = dataset.map(lambda x: tf.py_function(read_hdf5, [x], [tf.string, tf.float32]), num_parallel_calls=args.workers)
    # dataset = dataset.batch(args.batch, drop_remainder=False)


def write_data(args, filename, data):

    with h5py.File(os.path.join(args.result_path, filename.split("/")[-1] + ".h5"), "w") as f:
        f.create_dataset("Data", data=data.numpy())
