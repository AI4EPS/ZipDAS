# %%
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import multiprocessing as mp
import torch
import pandas as pd
from compdas.data import normalize, percentile_filter, equalize_hist, Config, read_hdf5

# %%
## Wavelet
# @dataclass
# class Config:
#     method: str = "wavelet"
#     keep_ratio: float = 0.1
#     quality: float = 10.0
#     plot_figure: bool = False

#     # data_path: str = "event_data"
#     # comp_path: str = "results/compressed_event"
#     # decomp_path: str = "results/decompressed_event"

#     data_path: str = "noise_data"
#     comp_path: str = "results/compressed_noise"
#     decomp_path: str = "results/decompressed_noise"

#     sensitivity = 6.769989802467751e-03

#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)


# @dataclass
# class Config:

#     data_format: str = "h5"
#     normalize_nt: int = 6000
#     normalize_nx: int = 1
#     sensitivity = 6.769989802467751e-03 # micro m/s for Redgecrest

#     workers: int = 1
#     percentile = 0.999

#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)


config = Config()

# %%
# def read_raw_h5(fname):
#     with h5py.File(fname, "r") as fp:
#         if "Data" in fp:
#             raw = fp["Data"][:]
#             raw = np.gradient(raw, axis=-1)
#         elif "data" in fp:
#             raw = fp["data"][:]
#         else:
#             raise ValueError(f"{raw} does not contain Data or data")

#     # max_mad = 6.0
#     # std = np.std(raw, axis=(-2, -1), keepdims=True)
#     # vmax = max_mad * std
#     # vmin = -vmax
#     # raw = raw * (raw > vmin) * (raw < vmax)

#     return raw

# def read_hdf5(args, filename):
#     """Loads a hdf5 file."""

#     with h5py.File(filename, "r") as f:

#         if "Data" in f:
#             data = f["Data"][:]  # (nc, nt)
#             data = np.gradient(data, f["Data"].attrs["dt"], axis=-1) * args.sensitivity
#         elif "data" in f:
#             data = f["data"][:].T
#         else:
#             raise ValueError(f"{filename} does not contain Data or data")

#         return {"data": data, "filename": filename}


# %%
def calc_mse(raw_list, processed_list):

    raw_list = sorted(raw_list)
    processed_list = sorted(processed_list)
    mse = []
    for raw, processed in zip(raw_list, processed_list):

        raw = read_hdf5(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = normalize(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = equalize_hist(config, raw)

        raw = raw["data"]

        nx_, nt_ = raw.shape
        # print(nx_, nt_)
        with h5py.File(processed, "r") as fp:
            processed = fp["data"][:nx_, :nt_]
            # print(f"{fp['data'].shape = }")
            # if "Data" in fp:
            #     processed = fp["Data"][:nx_, :nt_]
            #     processed = np.gradient(processed, axis=-1)
            # elif "data" in fp:
            #     processed = fp["data"][:].T
            #     processed = processed[:nx_, :nt_]

        
        raw = raw.astype(np.float32)
        processed = processed.astype(np.float32)

        # print(raw.max(), raw.min())
        # print(processed.max(), processed.min())
        # raise

        # mse.append(np.mean((raw - processed) ** 2) / np.mean(raw**2))
        # mse.append(np.mean((raw - processed) ** 2) / np.mean(raw**2))
        RMSD = np.sqrt(np.mean((raw - processed) ** 2)) / np.sqrt(np.mean(raw**2))
        # mse.append(RMSD / np.mean(np.abs(raw)))
        # mse.append(RMSD / np.mean(np.abs(raw)))
        mse.append(RMSD)

        # print(f"{np.mean((raw - processed)**2) = }")
        # print(raw.shape)
        # print(processed.shape)

        # raw = raw[:, 6000:6000+6000]
        # processed = processed[:, 6000:6000+6000]
        
        # raw -= np.mean(raw, axis=-1, keepdims=True)
        # raw /= np.std(raw, axis=-1, keepdims=True)
        # processed -= np.mean(processed, axis=-1, keepdims=True)
        # processed /= np.std(processed, axis=-1, keepdims=True)

        # vmax = 3
        # vmin = -vmax
        # print(f"{vmin = } {vmax = }")
        # plt.figure(figsize=(10, 4))
        # plt.imshow(raw, vmin=vmin, vmax=vmax, cmap="seismic", interpolation=None)
        # # plt.colorbar()
        # plt.savefig(f"raw.png", dpi=300, bbox_inches="tight")
        # plt.close()
        # plt.figure(figsize=(10, 4))
        # plt.imshow(processed, vmin=vmin, vmax=vmax, cmap="seismic", interpolation=None)
        # # plt.colorbar()
        # plt.savefig(f"processed.png", dpi=300, bbox_inches="tight")
        # # raise

        # # if np.mean((raw - processed)**2) > 0.01:
        # #     break

    return mse


def calc_mape(raw_list, processed_list):

    raw_list = sorted(raw_list)
    processed_list = sorted(processed_list)
    mape = []
    for raw, processed in zip(raw_list, processed_list):

        raw = read_hdf5(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = normalize(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = equalize_hist(config, raw)

        raw = raw["data"]
        # nx_, nt_ = raw.shape
        # print(nx_, nt_)

        nx_, nt_ = raw.shape
        with h5py.File(processed, "r") as fp:
            processed = fp["data"][:nx_, :nt_]
            # print(f"{fp['data'].shape = }")
        
        raw = raw.astype(np.float32)
        processed = processed.astype(np.float32)

        
        # mape.append(np.mean(np.abs((raw - processed)/raw)))
        mape.append(0)


    return mape

# %%
def calc_ssim(raw_list, processed_list):

    raw_list = sorted(raw_list)
    processed_list = sorted(processed_list)
    ssim = []
    for raw, processed in zip(raw_list, processed_list):

        raw = read_hdf5(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = normalize(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = equalize_hist(config, raw)

        raw = raw["data"]

        nx_, nt_ = raw.shape
        with h5py.File(processed, "r") as fp:
            processed = fp["data"][:nx_, :nt_]

        mu1 = np.mean(raw)
        mu2 = np.mean(processed)
        L = np.max(raw) - np.min(raw)
        k1 = 0.01
        k2 = 0.03
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        sigma1 = np.std(raw)
        sigma2 = np.std(processed)
        sigma12 = np.mean((raw - mu1) * (processed - mu2))
        ssim_ = (
            (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        )
        ssim.append(ssim_)

    return ssim


# %%
def calc_psnr(raw_list, processed_list):
    raw_list = sorted(raw_list)
    processed_list = sorted(processed_list)
    psnr = []
    for raw, processed in zip(raw_list, processed_list):
        raw = read_hdf5(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = normalize(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = equalize_hist(config, raw)

        raw = raw["data"]

        nx_, nt_ = raw.shape
        with h5py.File(processed, "r") as fp:
            processed = fp["data"][:nx_, :nt_]

        mse = np.mean((raw - processed) ** 2)
        psnr_ = 10 * np.log10(np.max(raw**2) / mse)
        psnr.append(psnr_)

    return psnr


# %%
def calc_compression_rate(raw_list, processed_list):
    raw_list = sorted(raw_list)
    processed_list = sorted(processed_list)
    raw_size, processed_size = [], []
    if len(raw_list) == len(processed_list):
        for raw, processed in zip(raw_list, processed_list):
            raw_size.append(os.path.getsize(raw))
            processed_size.append(os.path.getsize(processed))# + os.path.getsize(processed.with_suffix('.pkl')))
            # processed_size.append(os.path.getsize(processed))
        return np.array(raw_size) / np.array(processed_size)

    else:
        for raw in raw_list:
            raw_size.append(os.path.getsize(raw))
        for processed in processed_list:
            # processed_size.append(os.path.getsize(processed))# + os.path.getsize(processed.with_suffix('.pkl')))
            processed_size.append(os.path.getsize(processed))
        
        return [np.sum(raw_size) / np.sum(processed_size)]


# %%
def calc_cc(raw_list, processed_list):
    raw_list = sorted(raw_list)
    processed_list = sorted(processed_list)
    cc = []
    for raw, processed in zip(raw_list, processed_list):
        raw = read_hdf5(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = normalize(config, raw)
        # raw = percentile_filter(config, raw)
        # raw = equalize_hist(config, raw)

        raw = raw["data"]

        nx_, nt_ = raw.shape
        with h5py.File(processed, "r") as fp:
            processed = fp["data"][:nx_, :nt_]
        cc.append(np.corrcoef(raw.flatten(), processed.flatten())[0, 1])
    return cc


# %%
def benchmarking(data_type, method, keep_ratio, quality, compression_rate, mse, mape, ssim, psnr, corrcoef):

    ### Noise
    if data_type == "noise":
        
        config = Config(data_path = "noise_data", comp_path = "results/compressed_noise", decomp_path = "results/decompressed_noise", method=method, keep_ratio=keep_ratio, quality=quality, plot_figure=True)
        compress_cmd = f"python run.py --mode compress --data_path {config.data_path} --data_format h5 --result_path {config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f} --method={config.method} --keep_ratio={config.keep_ratio} --quality={config.quality}"
        if config.plot_figure:
            compress_cmd += " --plot_figure"
        print(compress_cmd)
        os.system(compress_cmd)

        decompress_cmd = f"python run.py --mode decompress --data_path {config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method} --result_path {config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f} --method={config.method}"
        if config.plot_figure:
            decompress_cmd += " --plot_figure"
        print(decompress_cmd)
        os.system(decompress_cmd)

        # cctorch_cmd = f"python CCTorch/run.py --data-list1=data.lst --data-path=noise_data --dt=0.04 --maxlag=30  --mode=AN  --block-size1 10 --block-size2 10 --fixed-channels 300 500 700 900 --result-path results/raw_cc_{config.keep_ratio:.2f}_{config.quality:.0f}"
        # os.system(cctorch_cmd)
        # cctorch_cmd = f"python CCTorch/run.py --data-list1=data.lst --data-path={config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method} --dt=0.04 --maxlag=30  --mode=AN  --block-size1 10 --block-size2 10 --fixed-channels 300 500 700 900 --result-path results/compressed_cc_{config.keep_ratio:.2f}_{config.quality:.0f}"
        # os.system(cctorch_cmd)  
        
        pass  

    ### Event
    if data_type == "event":

        config = Config(data_path = "event_data", comp_path = "results/compressed_event", decomp_path = "results/decompressed_event", method=method, keep_ratio=keep_ratio, quality=quality,  plot_figure=True)
        compress_cmd = f"python run.py --mode compress --data_path {config.data_path} --data_format h5 --result_path {config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f} --method={config.method} --keep_ratio={config.keep_ratio} --quality={config.quality}"
        if config.plot_figure:
            compress_cmd += " --plot_figure"
        print(compress_cmd)
        os.system(compress_cmd)

        decompress_cmd = f"python run.py --mode decompress --data_path {config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method} --result_path {config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f} --method={config.method}"
        if config.plot_figure:
            decompress_cmd += " --plot_figure"
        print(decompress_cmd)
        os.system(decompress_cmd)

    # %%
    raw_data = list(Path(config.data_path).glob("*.h5"))
    if config.method == "jpeg":
        compressed_data = list((Path(f"{config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}") / config.method).glob("*.j2k"))
    else:
        compressed_data = list((Path(f"{config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}") / config.method).glob("*.npz"))
    decomressed_data = list((Path(f"{config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}") / config.method).glob("*.h5"))

    raw_data_preprocess = []
    for raw in raw_data:

        raw = read_hdf5(config, raw)
        raw = percentile_filter(config, raw)
        raw = normalize(config, raw)
        raw = percentile_filter(config, raw)
        raw = equalize_hist(config, raw)

        with h5py.File(Path(f"results/preprocessed_{config.keep_ratio:.2f}_{config.quality:.0f}_preprocess.h5"), "w") as f:
            f.create_dataset("data", data=raw["data"].T)

        raw_data_preprocess.append(Path(f"results/preprocessed_{config.keep_ratio:.2f}_{config.quality:.0f}_preprocess.h5"))
    
    raw_data = raw_data_preprocess
    print(raw_data, compressed_data, decomressed_data)

    rate_ = calc_compression_rate(raw_data, compressed_data)
    compression_rate[f"{keep_ratio:.2f}"] = rate_
    compression_rate[f"{quality:.0f}"] = rate_

    mse_ = calc_mse(raw_data, decomressed_data)
    mse[f"{keep_ratio:.2f}"] = mse_
    mse[f"{quality:.0f}"] = mse_

    mape_ = calc_mape(raw_data, decomressed_data)
    mape[f"{keep_ratio:.2f}"] = mape_
    mape[f"{quality:.0f}"] = mape_

    ssim_ = calc_ssim(raw_data, decomressed_data)
    ssim[f"{keep_ratio:.2f}"] = ssim_
    ssim[f"{quality:.0f}"] = ssim_

    psnr_ = calc_psnr(raw_data, decomressed_data)
    psnr[f"{keep_ratio:.2f}"] = psnr_
    psnr[f"{quality:.0f}"] = psnr_

    cc_ = calc_cc(raw_data, decomressed_data)
    corrcoef[f"{keep_ratio:.2f}"] = cc_
    corrcoef[f"{quality:.0f}"] = cc_


def read_AN(data_type, method, keep_ratio, channels, files):

    data_list = []
    for f in files:
        with h5py.File(f, "r") as fp:
            for chn in channels:
                data = []
                index = []
                for c in sorted(fp[f"/{chn}"].keys(), key=lambda x: int(x.split("/")[-1])):
                    data.append(fp[f"/{chn}/{c}"]["xcorr"][:])
                    index.append(c)
                    # print(fp[f"/{chn}/{c}"]["xcorr"][:].shape)
                data = np.stack(data)
                # fig, axes = plt.subplots(1, 1)
                # vmax = np.std(data)
                # im = axes.imshow(data, vmin=-vmax, vmax=vmax, aspect="auto", cmap="RdBu")
                # fig.colorbar(im, ax=axes)
                # fig.savefig(f"result_{chn}_{method}_{data_type}_{keep_ratio:.2f}.png", dpi=300, bbox_inches="tight")

                data_list.append(data)

    return data_list

    
def benchmarking_AN(data_type, method, keep_ratio, compression_rate, mse, ssim, psnr, corrcoef):

    ### Noise
    if data_type == "noise":
        config = Config(data_path = "noise_data", comp_path = "results/compressed_noise", decomp_path = "results/decompressed_noise", method=method, keep_ratio=keep_ratio, plot_figure=True)
        # compress_cmd = f"python run.py --mode compress --data_path {config.data_path} --data_format h5 --result_path {config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f} --method={config.method} --keep_ratio={config.keep_ratio}"
        # if config.plot_figure:
        #     compress_cmd += " --plot_figure"
        # os.system(compress_cmd)

        # decompress_cmd = f"python run.py --mode decompress --data_path {config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method} --result_path {config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f} --method={config.method}"
        # if config.plot_figure:
        #     compress_cmd += " --plot_figure"
        # os.system(decompress_cmd)

        # cctorch_cmd = f"python CCTorch/run.py --data-list1=data.lst --data-path=noise_data --dt=0.04 --maxlag=30  --mode=AN  --block-size1 10 --block-size2 10 --fixed-channels 300 --result-path results/raw_cc_{config.keep_ratio:.2f}_{config.quality:.0f}"
        # os.system(cctorch_cmd)

        if method == "jpeg":
            cctorch_cmd = f"python CCTorch/run.py --data-list1=data.lst --data-path={config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method} --dt=0.04 --maxlag=30  --mode=AN  --format j2k --block-size1 10 --block-size2 10 --fixed-channels 300 --result-path results/compressed_cc_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method}"
        else:
            cctorch_cmd = f"python CCTorch/run.py --data-list1=data.lst --data-path={config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method} --dt=0.04 --maxlag=30  --mode=AN  --block-size1 10 --block-size2 10 --fixed-channels 300 --result-path results/compressed_cc_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method}"
        print(cctorch_cmd)
        os.system(cctorch_cmd)       

    # %%
    raw_data = list(Path(f"results/raw_cc").glob("*.h5"))
    compressed_data = list(Path(f"results/compressed_cc_{config.keep_ratio:.2f}_{config.quality:.0f}").glob("*.h5"))

    channels = [300]
    raw= read_AN(data_type, method, keep_ratio, channels, raw_data)
    processed = read_AN(data_type, method, keep_ratio, channels, compressed_data)

    raw = np.concatenate(raw, axis=0)
    processed = np.concatenate(processed, axis=0)

    mse[f"{keep_ratio:.2f}"] = [np.mean((raw - processed) ** 2) / np.mean(raw**2)]

    mu1 = np.mean(raw)
    mu2 = np.mean(processed)
    L = np.max(raw) - np.min(raw)
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    sigma1 = np.std(raw)
    sigma2 = np.std(processed)
    sigma12 = np.mean((raw - mu1) * (processed - mu2))
    ssim_ = (
        (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    )
    ssim[f"{keep_ratio:.2f}"] = [ssim_]

    mse = np.mean((raw - processed) ** 2)
    psnr_ = 10 * np.log10(np.max(raw**2) / mse)
    psnr[f"{keep_ratio:.2f}"] = [psnr_]

    corrcoef[f"{keep_ratio:.2f}"] = [np.corrcoef(raw.flatten(), processed.flatten())[0, 1]]


def plot_result(data_type, method, keep_ratios, compression_rate=None, mse=None, ssim=None, psnr=None, corrcoef=None, figure_path="results/figures"):

    fig, axs = plt.subplots(nrows=3, ncols=2, squeeze=False, figsize=(9, 9), sharex=True)

    if compression_rate is not None:
        bplot1 = axs[0, 0].boxplot(
            [compression_rate[f"{k:.2f}"] for k in keep_ratios],
            vert=True,
            patch_artist=True,
            labels=[f"{k:.2f}" for k in keep_ratios],
        )
        axs[0, 0].set_ylabel("Compression rate")

    bplot1 = axs[1, 0].boxplot(
        [mse[f"{k:.2f}"] for k in keep_ratios], vert=True, patch_artist=True, labels=[f"{k:.2f}" for k in keep_ratios]
    )
    axs[1, 0].set_ylabel("MSE")

    # axs[0,1].set_yscale('log')
    bplot1 = axs[1, 1].boxplot(
        [psnr[f"{k:.2f}"] for k in keep_ratios], vert=True, patch_artist=True, labels=[f"{k:.2f}" for k in keep_ratios]
    )
    axs[1, 1].set_ylabel("PSNR")

    bplot1 = axs[2, 0].boxplot(
        [ssim[f"{k:.2f}"] for k in keep_ratios], vert=True, patch_artist=True, labels=[f"{k:.2f}" for k in keep_ratios]
    )
    axs[2, 0].set_ylabel("SSIM")

    bplot1 = axs[2, 1].boxplot(
        [corrcoef[f"{k:.2f}"] for k in keep_ratios],
        vert=True,
        patch_artist=True,
        labels=[f"{k:.2f}" for k in keep_ratios],
    )
    axs[2, 1].set_ylabel("Correlation Coefficient")
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, f"{data_type}_{method}.png"), dpi=300)
    fig.savefig(os.path.join(figure_path, f"{data_type}_{method}.pdf"), dpi=300)


# def plot_result_quality(data_type, method, qualities, compression_rate=None, mse=None, ssim=None, psnr=None, corrcoef=None, figure_path="results/figures"):

#     fig, axs = plt.subplots(nrows=3, ncols=2, squeeze=False, figsize=(9, 9), sharex=True)

#     if compression_rate is not None:
#         bplot1 = axs[0, 0].boxplot(
#             [compression_rate[f"{k:.0f}"] for k in qualities],
#             vert=True,
#             patch_artist=True,
#             labels=[f"{k:.0f}" for k in qualities],
#         )
#         axs[0, 0].set_ylabel("Compression rate")

#     bplot1 = axs[1, 0].boxplot(
#         [mse[f"{k:.0f}"] for k in qualities], vert=True, patch_artist=True, labels=[f"{k:.0f}" for k in qualities]
#     )
#     axs[1, 0].set_ylabel("MSE")

#     # axs[0,1].set_yscale('log')
#     bplot1 = axs[1, 1].boxplot(
#         [psnr[f"{k:.0f}"] for k in qualities], vert=True, patch_artist=True, labels=[f"{k:.0f}" for k in qualities]
#     )
#     axs[1, 1].set_ylabel("PSNR")

#     bplot1 = axs[2, 0].boxplot(
#         [ssim[f"{k:.0f}"] for k in qualities], vert=True, patch_artist=True, labels=[f"{k:.0f}" for k in qualities]
#     )
#     axs[2, 0].set_ylabel("SSIM")

#     bplot1 = axs[2, 1].boxplot(
#         [corrcoef[f"{k:.0f}"] for k in qualities],
#         vert=True,
#         patch_artist=True,
#         labels=[f"{k:.0f}" for k in qualities],
#     )
#     axs[2, 1].set_ylabel("Correlation Coefficient")
#     plt.tight_layout()
#     fig.savefig(os.path.join(figure_path, f"{data_type}_{method}.png"), dpi=300)

def plot_result_quality(data_type, method, qualities, compression_rate=None, mse=None, mape=None, ssim=None, psnr=None, corrcoef=None, figure_path="results/figures"):

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 6), sharex=True)
    compression_rate = [compression_rate[f"{k:.0f}"] for k in qualities]
    mse = [mse[f"{k:.0f}"] for k in qualities]
    mape = [mape[f"{k:.0f}"] for k in qualities]

    # axs[0, 0].scatter(compression_rate, mape)
    # # axs[0, 0].set_xlabel("Compression rate")
    # axs[0, 0].set_ylabel("MAPE")
    print(f"{compression_rate = }")
    print(f"{mse = }")
    axs[0, 0].scatter(compression_rate, mse)
    # axs[0, 0].set_xlabel("Compression rate")
    axs[0, 0].set_ylabel("MSE")
    axs[0, 0].set_xscale('log')
    axs[0, 0].grid(True)

    psnr = [psnr[f"{k:.0f}"] for k in qualities]

    print(f"{psnr = }")
    axs[0, 1].scatter(compression_rate, psnr)
    # axs[0, 1].set_xlabel("Compression rate")
    axs[0, 1].set_ylabel("PSNR")
    axs[0, 1].set_xscale('log')
    axs[0, 1].grid(True)

    ssim = [ssim[f"{k:.0f}"] for k in qualities]
    print(f"{ssim = }")
    axs[1, 0].scatter(compression_rate, ssim)
    axs[1, 0].set_xlabel("Compression rate")
    axs[1, 0].set_ylabel("SSIM")
    axs[1, 0].set_xscale('log')
    axs[1, 0].grid(True)

    corrcoef = [corrcoef[f"{k:.0f}"] for k in qualities]
    print(f"{corrcoef = }")
    axs[1, 1].scatter(compression_rate, corrcoef)
    axs[1, 1].set_xlabel("Compression rate")
    axs[1, 1].set_ylabel("Correlation Coefficient")
    axs[1, 1].set_xscale('log')
    axs[1, 1].grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, f"{data_type}_{method}.png"), dpi=300)
    fig.savefig(os.path.join(figure_path, f"{data_type}_{method}.pdf"), dpi=300)

if __name__ == "__main__":

    keep_ratios = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    qualities = [1, 5, 10, 20, 40, 80]
    keep_ratio = 1.0
    quality = 1.0
    # keep_ratios = [0.02]
    # keep_ratios = [0.32]
    # ncpu = len(keep_ratios)
    # ncpu = 1
    # qualities = [1, 4, 10, 20, 40, 80][::-1]
    qualities = [2, 5, 11, 21, 41, 81][::-1]
    # qualities = [80]

    ## Wavelet
    # data_type = "event"
    data_type = "noise"
    if data_type == "event":
        ncpu = len(keep_ratios)
    if data_type == "noise":
        ncpu = 1
        # ncpu = 2
    # method = "curvelet"
    method = "wavelet"

    method = "jpeg"
    if data_type == "event":
        ncpu = len(qualities)
    if data_type == "noise":
        ncpu = 1
    
    
    # with mp.Manager() as manager:
    #     compression_rate = manager.dict()
    #     mse = manager.dict()
    #     mape = manager.dict()
    #     ssim = manager.dict()
    #     psnr = manager.dict()
    #     corrcoef = manager.dict()
    #     with mp.get_context("spawn").Pool(ncpu) as p:
    #         if method == "jpeg":
    #             p.starmap(
    #                 benchmarking,
    #                 [(data_type, method, keep_ratio, quality, compression_rate, mse, mape, ssim, psnr, corrcoef) for quality in qualities],
    #             )   
    #         else:
    #             p.starmap(
    #                 benchmarking,
    #                 [(data_type, method, keep_ratio, quality, compression_rate, mse, mape, ssim, psnr, corrcoef) for keep_ratio in keep_ratios],
    #             )
    #     compression_rate = dict(compression_rate)
    #     mse = dict(mse)
    #     mape = dict(mape)
    #     ssim = dict(ssim)
    #     psnr = dict(psnr)
    #     corrcoef = dict(corrcoef)

    # figure_path: str = "results/figures"
    # if not os.path.exists(figure_path):
    #     os.makedirs(figure_path)
    # if method == "jpeg":
    #     plot_result_quality(data_type, method, qualities, compression_rate, mse, mape, ssim, psnr, corrcoef, figure_path)
    # else:
    #     plot_result(data_type, method, keep_ratios, compression_rate, mse, mape, ssim, psnr, corrcoef, figure_path)
    


    if data_type == "noise":
        
        # if method == "jpeg":
        #     cctorch_cmd = f"python CCTorch/run.py --data-list1=data.lst --data-path=noise_data --dt=0.04 --maxlag=30  --mode=AN  --format j2k --block-size1 10 --block-size2 10 --fixed-channels 300 --result-path results/raw_cc"
        # else:
        # cctorch_cmd = f"python CCTorch/run.py --data-list1=data.lst --data-path=noise_data --dt=0.04 --maxlag=30  --mode=AN  --block-size1 10 --block-size2 10 --fixed-channels 300 --result-path results/raw_cc"
        cctorch_cmd = f"python CCTorch/run.py --data_list1=preprocessed.lst --data_path=results --dt=0.04 --maxlag=30  --mode=AN  --block_size1 10 --block_size2 10 --fixed_channels 300 --result_path results/raw_cc"
        print(cctorch_cmd)
        # raise
        os.system(cctorch_cmd)

        with mp.Manager() as manager:
            compression_rate = manager.dict()
            mse = manager.dict()
            ssim = manager.dict()
            psnr = manager.dict()
            corrcoef = manager.dict()
            with mp.get_context("spawn").Pool(ncpu) as p:
                p.starmap(
                    benchmarking_AN,
                    [(data_type, method, keep_ratio, compression_rate, mse, ssim, psnr, corrcoef) for keep_ratio in keep_ratios],
                )
            compression_rate = dict(compression_rate)
            mse = dict(mse)
            ssim = dict(ssim)
            psnr = dict(psnr)
            corrcoef = dict(corrcoef)

        figure_path: str = "results/figures_AN"
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plot_result(data_type, method, keep_ratios, None, mse, ssim, psnr, corrcoef, figure_path)


    ## Curvelet
    # method = "curvelet"
    # with mp.Manager() as manager:
    #     compression_rate = manager.dict()
    #     mse = manager.dict()
    #     ssim = manager.dict()
    #     psnr = manager.dict()
    #     corrcoef = manager.dict()
    #     with mp.get_context("spawn").Pool(ncpu) as p:
    #         p.starmap(
    #             benchmarking,
    #             [(method, keep_ratio, compression_rate, mse, ssim, psnr, corrcoef) for keep_ratio in keep_ratios],
    #         )
    #     compression_rate = dict(compression_rate)
    #     mse = dict(mse)
    #     ssim = dict(ssim)
    #     psnr = dict(psnr)
    #     corrcoef = dict(corrcoef)

    # figure_path: str = "results/figures"
    # if not os.path.exists(figure_path):
    #     os.makedirs(figure_path)
    # plot_result(method, keep_ratios, compression_rate, mse, ssim, psnr, corrcoef, figure_path)

## Neural Network
# %%
