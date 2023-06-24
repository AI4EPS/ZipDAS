# %%
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from zipdas.data import read_hdf5, Config
import json
import os

# %%
config = Config()
file_name = "Ridgecrest_ODH3-2021-06-15 183838Z.h5"
raw_path = Path("noise_data/")
# preprocess_path = Path("results/compressed_noise/jpeg/preprocess/")
preprocess_path = Path("results/compressed_noise/jpeg/preprocess/")
preprocess_cc_path = Path("results/cctorch_noise/")

# %%
# with h5py.File(raw_path / file_name, "r") as f:
#     data = f["Data"][:].T
#     data = np.gradient(data, f["Data"].attrs["dt"], axis=-1) * config.sensitivity

# %%
with h5py.File(preprocess_path / file_name, "r") as f:
    raw = f["data"][:]
raw_size = os.path.getsize(preprocess_path / file_name)

raw_cc = []
for f in preprocess_cc_path.glob("*.npz"):
    data = np.load(f)["data"]
    raw_cc.append(data)
raw_cc = np.stack(raw_cc)

def calc_mse(raw, data):
    print(f"{raw.std() = }, {data.std() = }")
    # raise
    mse = np.mean((raw - data) ** 2)
    return mse

def calc_mad(raw, data):
    mse = np.mean(np.abs(raw - data))
    return mse

def calc_rmsd(raw, data):
    RMSD = np.sqrt(np.mean((raw - data) ** 2)) / np.sqrt(np.mean(raw**2))
    return RMSD

def calc_ssim(raw, data):
    SSIM = ssim(raw, data, data_range=raw.max() - raw.min())
    return SSIM

def calc_psnr(raw, data):
    PSNR = 10 * np.log10(np.max(raw) ** 2 / np.mean((raw - data) ** 2))
    return PSNR

def calc_cc(raw, data):
    cc = np.corrcoef(raw.flatten(), data.flatten())[0, 1]
    return cc

def process(raw, raw_size, data_type, method, keep_ratio, quality):

    nx, nt = raw.shape
    decompress_path = Path(f"results/decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/")
    with h5py.File(decompress_path / file_name, "r") as f:
        data = f["data"][:nx, :nt]

    mse = calc_mse(raw, data)
    mad = calc_mad(raw, data)
    rmsd = calc_rmsd(raw, data)
    ssim = calc_ssim(raw, data)
    psnr = calc_psnr(raw, data)
    cc = calc_cc(raw, data)

    compress_path = f"results/compressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/{file_name}"
    compress_size = 0
    for f in os.listdir(compress_path):
        f = os.path.join(compress_path, f)
        if os.path.isfile(f):
            compress_size += os.path.getsize(f)    
    compression_rate =  raw_size / compress_size

    metrics = {
        "mse": mse.item(),
        "mad": mad.item(),
        "rmsd": rmsd.item(),
        "ssim": ssim.item(),
        "psnr": psnr.item(),
        "cc": cc.item(),
        "compression_rate": compression_rate,
        "data_type": data_type,
        "method": method,
        "keep_ratio": keep_ratio,
        "quality": quality,
    }
    print(f"decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/metrics.json", metrics)
    with open(f"results/decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def process_neural(raw, raw_size, data_type, method):

    nx, nt = raw.shape
    decompress_path = Path(f"results/decompressed_{data_type}/{method}/")
    with h5py.File(decompress_path / file_name, "r") as f:
        data = f["data"][:nx, :nt]

    mse = calc_mse(raw, data)
    mad = calc_mad(raw, data)
    rmsd = calc_rmsd(raw, data)
    ssim = calc_ssim(raw, data)
    psnr = calc_psnr(raw, data)
    cc = calc_cc(raw, data)

    compress_path = f"results/compressed_{data_type}/{method}/{file_name}"
    compress_size = 0
    for f in os.listdir(compress_path):
        f = os.path.join(compress_path, f)
        if os.path.isfile(f):
            compress_size += os.path.getsize(f)    
    compression_rate =  raw_size / compress_size

    metrics = {
        "mse": mse.item(),
        "mad": mad.item(),
        "rmsd": rmsd.item(),
        "ssim": ssim.item(),
        "psnr": psnr.item(),
        "cc": cc.item(),
        "compression_rate": compression_rate,
        "data_type": data_type,
        "method": method,
    }
    print(f"decompressed_{data_type}/{method}/metrics.json", metrics)
    with open(f"results/decompressed_{data_type}/{method}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def process_cc(raw, raw_size, raw_cc, data_type, method, keep_ratio, quality):

    nx, nt = raw.shape
    decompress_path = Path(f"results/decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/cctorch/")
    data_cc = []
    for f in decompress_path.glob("*.npz"):
        data = np.load(f)["data"]
        data_cc.append(data)
    data_cc = np.stack(data_cc)

    compress_path = f"results/compressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/{file_name}"
    compress_size = 0
    for f in os.listdir(compress_path):
        f = os.path.join(compress_path, f)
        if os.path.isfile(f):
            compress_size += os.path.getsize(f)    
    compression_rate_ =  raw_size / compress_size

    mse = []
    mad = []
    rmsd = []
    ssim = []
    psnr = []
    cc = []
    compression_rate = []
    for i in range(len(data_cc)):
        mse_ = calc_mse(raw_cc[i], data_cc[i])
        mad_ = calc_mad(raw_cc[i], data_cc[i])
        rmsd_ = calc_rmsd(raw_cc[i], data_cc[i])
        ssim_ = calc_ssim(raw_cc[i], data_cc[i])
        psnr_ = calc_psnr(raw_cc[i], data_cc[i])
        cc_ = calc_cc(raw_cc[i], data_cc[i])
        mse.append(mse_.item())
        mad.append(mad_.item())
        rmsd.append(rmsd_.item())
        ssim.append(ssim_.item())
        psnr.append(psnr_.item())
        cc.append(cc_.item())
        compression_rate.append(compression_rate_)

    metrics = {
        "mse": mse,
        "mad": mad,
        "rmsd": rmsd,
        "ssim": ssim,
        "psnr": psnr,
        "cc": cc,
        "compression_rate": compression_rate,
        "data_type": data_type,
        "method": method,
        "keep_ratio": keep_ratio,
        "quality": quality,
    }
    print(f"decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/metrics_cc.json", metrics)
    with open(f"results/decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/metrics_cc.json", "w") as f:
        json.dump(metrics, f, indent=4)


def process_cc_neural(raw, raw_size, raw_cc, data_type, method):

    nx, nt = raw.shape
    decompress_path = Path(f"results/decompressed_{data_type}/{method}/cctorch/")
    data_cc = []
    for f in decompress_path.glob("*.npz"):
        data = np.load(f)["data"]
        data_cc.append(data)
    data_cc = np.stack(data_cc)

    compress_path = f"results/compressed_{data_type}/{method}/{file_name}"
    compress_size = 0
    for f in os.listdir(compress_path):
        f = os.path.join(compress_path, f)
        if os.path.isfile(f):
            compress_size += os.path.getsize(f)    
    compression_rate_ =  raw_size / compress_size

    mse = []
    mad = []
    rmsd = []
    ssim = []
    psnr = []
    cc = []
    compression_rate = []
    for i in range(len(data_cc)):
        mse_ = calc_mse(raw_cc[i], data_cc[i])
        mad_ = calc_mad(raw_cc[i], data_cc[i])
        rmsd_ = calc_rmsd(raw_cc[i], data_cc[i])
        ssim_ = calc_ssim(raw_cc[i], data_cc[i])
        psnr_ = calc_psnr(raw_cc[i], data_cc[i])
        cc_ = calc_cc(raw_cc[i], data_cc[i])
        mse.append(mse_.item())
        mad.append(mad_.item())
        rmsd.append(rmsd_.item())
        ssim.append(ssim_.item())
        psnr.append(psnr_.item())
        cc.append(cc_.item())
        compression_rate.append(compression_rate_)

    metrics = {
        "mse": mse,
        "mad": mad,
        "rmsd": rmsd,
        "ssim": ssim,
        "psnr": psnr,
        "cc": cc,
        "compression_rate": compression_rate,
        "data_type": data_type,
        "method": method,
    }
    print(f"decompressed_{data_type}/{method}/metrics_cc.json", metrics)
    with open(f"results/decompressed_{data_type}/{method}/metrics_cc.json", "w") as f:
        json.dump(metrics, f, indent=4)

# %%
keep_ratios = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 1.0]
# keep_ratios = [1.0]
# qualities = [1, 4, 10, 20, 40, 80]
qualities = [1, 2, 4, 7, 10, 20, 40, 80]
data_types = ["noise"]
methods = ["jpeg", "wavelet", "curvelet", "neural"]
methods = ["jpeg", "neural"]
methods = ["jpeg"]
ncpu = 1
calculate_metrics = False

if calculate_metrics:
    
    methods = ["jpeg", "neural"]

    for method in methods:
        if method == "jpeg":
            keep_ratio = 1.0
            for quality in qualities:
                for data_type in data_types:
                    process(raw, raw_size, data_type, method, keep_ratio, quality)
        elif method == "neural":
            for data_type in data_types:
                process_neural(raw, raw_size, data_type, method)
        else:
            quality = 1.0
            for keep_ratio in keep_ratios:
                for data_type in data_types:
                    process(raw, raw_size, data_type, method, keep_ratio, quality)

    # %%
    for method in methods:
        if method == "jpeg":
            keep_ratio = 1.0
            for quality in qualities:
                for data_type in data_types:
                    process_cc(raw, raw_size, raw_cc, data_type, method, keep_ratio, quality)
        elif method == "neural":
            for data_type in data_types:
                process_cc_neural(raw, raw_size, raw_cc, data_type, method)
        else:
            quality = 1.0
            for keep_ratio in keep_ratios:
                for data_type in data_types:
                    process_cc(raw, raw_size, raw_cc, data_type, method, keep_ratio, quality)

# %%
methods = ["jpeg", "neural"]

for data_type in data_types:
    metrics = {}
    for method in methods:
        if method == "jpeg":
            keep_ratio = 1.0
            x = []
            y = []
            for quality in qualities:
                with open(f"results/decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/metrics.json", "r") as f:
                    tmp = json.load(f)
                    x.append(tmp["compression_rate"])
                    y.append([tmp["mse"], tmp["mad"], tmp["rmsd"], tmp["psnr"], tmp["cc"], tmp["ssim"]])
            metrics[f"{method}"] = [x, y]
        elif method == "neural":
            x = []
            y = []
            with open(f"results/decompressed_{data_type}/{method}/metrics.json", "r") as f:
                tmp = json.load(f)
                x.append(tmp["compression_rate"])
                y.append([tmp["mse"], tmp["mad"], tmp["rmsd"], tmp["psnr"],  tmp["cc"], tmp["ssim"]])
            metrics[f"{method}"] = [x, y]
        else:
            quality = 1.0
            x = []
            y = []
            for keep_ratio in keep_ratios:
                with open(f"results/decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/metrics.json", "r") as f:
                    tmp = json.load(f)
                    x.append(tmp["compression_rate"])
                    y.append([tmp["mse"], tmp["mad"], tmp["rmsd"], tmp["psnr"],  tmp["cc"], tmp["ssim"]])
            metrics[f"{method}"] = [x, y]
        

# %%
for data_type in data_types:
    metrics_cc = {}
    for method in methods:
        if method == "jpeg":
            keep_ratio = 1.0
            x = []
            y = []
            for quality in qualities:
                with open(f"results/decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/metrics_cc.json", "r") as f:
                    tmp = json.load(f)
                    x.append(tmp["compression_rate"])
                    y.append([tmp["mse"], tmp["mad"], tmp["rmsd"], tmp["psnr"], tmp["cc"], tmp["ssim"]])
            metrics_cc[f"{method}"] = [x, y]
        elif method == "neural":
            x = []
            y = []
            with open(f"results/decompressed_{data_type}/{method}/metrics_cc.json", "r") as f:
                tmp = json.load(f)
                x.append(tmp["compression_rate"])
                y.append([tmp["mse"], tmp["mad"], tmp["rmsd"], tmp["psnr"],  tmp["cc"], tmp["ssim"]])
            metrics_cc[f"{method}"] = [x, y]
        else:
            quality = 1.0
            x = []
            y = []
            for keep_ratio in keep_ratios:
                with open(f"results/decompressed_{data_type}_{keep_ratio:.2f}_{quality:.0f}/{method}/metrics_cc.json", "r") as f:
                    tmp = json.load(f)
                    x.append(tmp["compression_rate"])
                    y.append([tmp["mse"], tmp["mad"], tmp["rmsd"], tmp["psnr"],  tmp["cc"], tmp["ssim"]])
            metrics_cc[f"{method}"] = [x, y]

# %%
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

for i, method in enumerate(methods):
    axs[0, 0].scatter(metrics[method][0], [y[2] for y in metrics[method][1]], c=f"C{i}", label=method.upper())
    axs[0, 0].plot(metrics[method][0], [y[2] for y in metrics[method][1]], c=f"C{i}")
    axs[0, 0].set_xlabel("Compression rate")
    axs[0, 0].set_ylabel("RMSD")
    axs[0, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    axs[0, 1].scatter(metrics[method][0], [y[3] for y in metrics[method][1]], c=f"C{i}", label=method.upper())
    axs[0, 1].plot(metrics[method][0], [y[3] for y in metrics[method][1]], c=f"C{i}")
    axs[0, 1].set_xlabel("Compression rate")
    axs[0, 1].set_ylabel("PSNR")
    axs[0, 1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    axs[1, 0].scatter(metrics[method][0], [y[4] for y in metrics[method][1]], c=f"C{i}", label=method.upper())
    axs[1, 0].plot(metrics[method][0], [y[4] for y in metrics[method][1]], c=f"C{i}")
    axs[1, 0].set_xlabel("Compression rate")
    axs[1, 0].set_ylabel("Cross-correlation")
    axs[1, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    axs[1, 1].scatter(metrics[method][0], [y[5] for y in metrics[method][1]], c=f"C{i}", label=method.upper())
    axs[1, 1].plot(metrics[method][0], [y[5] for y in metrics[method][1]], c=f"C{i}")
    axs[1, 1].set_xlabel("Compression rate")
    axs[1, 1].set_ylabel("SSIM")
    axs[1, 1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)


axs[0, 0].legend(loc="lower right")
axs[0, 0].set_xlim(left=1)
axs[0, 0].set_xscale("log")
axs[0, 1].legend(loc="upper right")
axs[0, 1].set_xscale("log")
axs[0, 1].set_xlim(left=1)
axs[1, 0].legend(loc="lower left")
axs[1, 0].set_xscale("log")
axs[1, 0].set_xlim(left=1)
axs[1, 1].legend(loc="lower left")
axs[1, 1].set_xscale("log")
axs[1, 1].set_xlim(left=1)
fig.tight_layout()
fig.savefig("results/metrics.png", dpi=300, bbox_inches="tight")
fig.savefig("results/metrics.pdf", dpi=300, bbox_inches="tight")


# %%
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

for i, method in enumerate(methods):
    axs[0, 0].scatter(metrics_cc[method][0], [y[2] for y in metrics_cc[method][1]], s=20, c=f"C{i}", label=method.upper())
    axs[0, 0].plot(metrics_cc[method][0], [y[2] for y in metrics_cc[method][1]], c=f"C{i}", linestyle="--", linewidth=1.0)
    axs[0, 0].set_xlabel("Compression rate")
    axs[0, 0].set_ylabel("RMSD")
    axs[0, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    axs[0, 1].scatter(metrics_cc[method][0], [y[3] for y in metrics_cc[method][1]], s=20, c=f"C{i}", label=method.upper())
    axs[0, 1].plot(metrics_cc[method][0], [y[3] for y in metrics_cc[method][1]], c=f"C{i}", linestyle="--", linewidth=1.0)
    axs[0, 1].set_xlabel("Compression rate")
    axs[0, 1].set_ylabel("PSNR")
    axs[0, 1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    axs[1, 0].scatter(metrics_cc[method][0], [y[4] for y in metrics_cc[method][1]], s=20, c=f"C{i}", label=method.upper())
    axs[1, 0].plot(metrics_cc[method][0], [y[4] for y in metrics_cc[method][1]], c=f"C{i}", linestyle="--", linewidth=1.0)
    axs[1, 0].set_xlabel("Compression rate")
    axs[1, 0].set_ylabel("Cross-correlation")
    axs[1, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    axs[1, 1].scatter(metrics_cc[method][0], [y[5] for y in metrics_cc[method][1]], s=20, c=f"C{i}", label=method.upper())
    axs[1, 1].plot(metrics_cc[method][0], [y[5] for y in metrics_cc[method][1]], c=f"C{i}", linestyle="--", linewidth=1.0)
    axs[1, 1].set_xlabel("Compression rate")
    axs[1, 1].set_ylabel("SSIM")
    axs[1, 1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)


axs[0, 0].legend(loc="lower right")
axs[0, 0].set_xlim(left=1)
axs[0, 0].set_xscale("log")
axs[0, 1].legend(loc="upper right")
axs[0, 1].set_xscale("log")
axs[0, 1].set_xlim(left=1)
axs[1, 0].legend(loc="lower left")
axs[1, 0].set_xscale("log")
axs[1, 0].set_xlim(left=1)
axs[1, 1].legend(loc="lower left")
axs[1, 1].set_xscale("log")
axs[1, 1].set_xlim(left=1)
fig.tight_layout()
fig.savefig("results/metrics_cc.png", dpi=300, bbox_inches="tight")
fig.savefig("results/metrics_cc.pdf", dpi=300, bbox_inches="tight")