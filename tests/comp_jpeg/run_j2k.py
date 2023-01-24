import numpy as np
import h5py
import imageio.v3 as iio
import io
import glob
from dataclasses import dataclass
import os
from tqdm import tqdm

def read_data(filename):

    with h5py.File(filename, "r") as fp:
        data = fp["Data"][:]

    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)

    # return data[:, :203264]
    # return data[:, :102400]
    return data


def data_to_j2k(data, filename="data_compressed.rb", nt=10240):

    h, w = data.shape
    for i in tqdm(range(0, w, nt)):
        tmp = data[:, i : i + nt]

        meta = io.BytesIO()
        iio.imwrite(meta, tmp, plugin="pillow", extension=".j2k", quality_mode="rates", quality_layers=[30], irreversible=True)
        
        with open(filename+f"_{i//nt:02d}", "wb") as f:
            f.write(meta.getbuffer())

def j2k_to_data(filename="data_compressed.rb", nt=10240):

    filenames = sorted(glob.glob(filename+"_*"))
    data_list = []
    for file in filenames:
        print(file)
        with open(file, "rb") as f:
            meta = io.BytesIO(f.read())
            data = iio.imread(meta, plugin="pillow", extension=".j2k")
            data_list.append(data)

    data = np.concatenate(data_list, axis=-1)
    with h5py.File(filename.replace(".rb", ".h5"), "w") as f:
        f.create_dataset("Data", data=np.float32(data))

    return data

@dataclass
class Args:
    input_path: str = "data"
    output_path: str = "data_compressed"
    format: str = "h5"

if __name__ == "__main__":

    args = Args()

    files = glob.glob(args.input_path + "/*." + args.format)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    for f in files:
        raw_data = read_data(f)
        print("Raw data:", raw_data.shape,  raw_data.dtype)

        output_filename = os.path.join(args.output_path, os.path.basename(f).replace(args.format, "rb"))
        data_to_j2k(raw_data, filename=output_filename)

        compressed_data = j2k_to_data(filename=output_filename)
        print("Compressed data:", compressed_data.shape, compressed_data.dtype)