import numpy as np
import h5py
import imageio.v3 as iio
import io
import glob
from dataclasses import dataclass
import os
from tqdm import tqdm

def normalize(data, **kwargs):
    vmin = np.min(data)
    vmax = np.max(data)
    data = (data - vmin) / (vmax - vmin) * 255
    return data, vmin, vmax

def denormalize(data, vmin, vmax, **kwargs):
    data = data / 255.0 * (vmax - vmin) + vmin
    return data

def read_data(filename):

    with h5py.File(filename, "r") as fp:
        data = fp["Data"][:]

    # vmin = np.min(data)
    # vmax = np.max(data)
    # data = (data - vmin) / (vmax - vmin) * 255
    data, vmin, vmax = normalize(data)
    data = data.astype(np.uint8)

    # return data[:, :203264]
    # return data[:, :102400]
    return {"data": data, "vmin": vmin, "vmax": vmax}


def data_to_j2k(data, filename="data_compressed.rb", nt=10240, args=None):

    h, w = data.shape
    quality = args.quality
    for i in tqdm(range(0, w, nt), desc="Compressing"):
        tmp = data[:, i : i + nt]

        meta = io.BytesIO()
        iio.imwrite(meta, tmp, plugin="pillow", extension=".j2k", quality_mode="rates", quality_layers=[quality], irreversible=True)
        
        with open(filename+f"_{i//nt:02d}", "wb") as f:
            f.write(meta.getbuffer())

def j2k_to_data(filename="data_compressed.rb", nt=10240, vmin=0.0, vmax=255.0, args=None):

    filenames = sorted(glob.glob(filename+"_*"))
    data_list = []
    for file in tqdm(filenames, desc="Decompressing"):
        with open(file, "rb") as f:
            meta = io.BytesIO(f.read())
            data = iio.imread(meta, plugin="pillow", extension=".j2k")
            data_list.append(data)

    data = np.concatenate(data_list, axis=-1)
    # data_float = (np.float32(data)/255.0 * (vmax - vmin) + vmin)
    data = np.float32(data)
    data = denormalize(data, vmin=vmin, vmax=vmax)
    # data = data - np.mean(data)
    with h5py.File(filename.replace(".rb", ".h5"), "w") as f:
        f.create_dataset("Data", data=data)

    return data

@dataclass
class Args:
    quality: int = 10
    input_path: str = "data"
    output_path: str = f"data_compressed_{quality:d}"
    format: str = "h5"
    

if __name__ == "__main__":

    args = Args()

    files = glob.glob(args.input_path + "/*." + args.format)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    for f in files:
        meta = read_data(f)
        print("Raw data:", meta["data"].shape,  meta["data"].dtype)

        output_filename = os.path.join(args.output_path, os.path.basename(f).replace(args.format, "rb"))
        data_to_j2k(meta["data"], filename=output_filename, args=args)

        compressed_data = j2k_to_data(filename=output_filename, vmin=meta["vmin"], vmax=meta["vmax"], args=args)
        print("Compressed data:", compressed_data.shape, compressed_data.dtype)