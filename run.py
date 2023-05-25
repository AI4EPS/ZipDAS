import argparse
import json
import os
from dataclasses import dataclass

from zipdas import (
    compress,
    compress_jpeg,
    compress_neural,
    decompress,
    decompress_jpeg,
    decompress_neural,
    download_and_unzip,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="Neural", help="Neural, Wavelet, Curvelet")
    parser.add_argument("--mode", type=str, default="compress", help="compress or decompress")
    parser.add_argument("--model_path", type=str, default="model.h5", help="model path")
    parser.add_argument("--data_path", type=str, default="data", help="data path")
    parser.add_argument("--result_path", type=str, default="results", help="result path")
    parser.add_argument("--data_format", type=str, default="h5", help="data data_format")
    parser.add_argument("--keep_ratio", type=float, default=0.1, help="keep ratio")
    parser.add_argument("--quality", type=float, default=50, help="quality")
    parser.add_argument("--save_preprocess", action="store_true", help="save reprocess data")
    parser.add_argument("--recover_amplitude", action="store_true", help="recover amplitude")
    parser.add_argument("--config", type=str, default="config.json", help="config file")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--batch_nt", type=int, default=360000, help="number of time samples")
    parser.add_argument("--plot_figure", action="store_true", help="plot figure")
    parser.add_argument("--quantile_filter", action="store_true", help="quantile filter")
    parser.add_argument("--workers", type=int, default=1, help="number of threads for preprocessing")
    args = parser.parse_args()
    return args


@dataclass
class Config:
    batch_nt: int = 6000
    config: str = "config.json"
    data_format: str = "h5"
    data_path: str = "data"
    figure_path: str = "figures"
    keep_ratio: float = 0.1
    method: str = "jpeg"
    model_path: str = "model.h5"
    mode: str = "compress"
    preprocess_path: str = "preprocess"
    plot_figure: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def main(args):

    if args.method == "Neural":
        import tensorflow as tf
        import tensorflow_compression as tfc

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {}
    config.update(vars(args))
    args = Config(**config)
    print(json.dumps(vars(args), indent=4, sort_keys=True))

    args.result_path = os.path.join(args.result_path, args.method)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if args.plot_figure:
        args.figure_path = os.path.join(args.result_path, "figures")
        if not os.path.exists(args.figure_path):
            os.makedirs(args.figure_path)

    if args.save_preprocess:
        args.preprocess_path = os.path.join(args.result_path, "preprocess")
        if not os.path.exists(args.preprocess_path):
            os.makedirs(args.preprocess_path)

    if args.mode == "compress":

        if args.method == "Neual":
            if not os.path.exists(args.model_path):
                model_url = "https://github.com/AI4EPS/models/releases/download/CompDAS-v2/model.zip"
                download_and_unzip(args.model_path)
            model = tf.keras.models.load_model(args.model_path)
            dtypes = [t.dtype for t in model.decompress.input_signature]
            compress_neural(args, model, dtypes)

        elif args.method == "jpeg":
            compress_jpeg(args)

        elif (args.method == "wavelet") or (args.method == "curvelet"):
            compress(args)

        else:
            raise ("method must be neural, wavelet, curvelet, or jpeg")

    elif args.mode == "decompress":

        if args.method == "Neual":
            model = tf.keras.models.load_model(args.model_path)
            dtypes = [t.dtype for t in model.decompress.input_signature]
            decompress_neural(args, model, dtypes)

        elif args.method == "jpeg":
            decompress_jpeg(args)

        elif (args.method == "wavelet") or (args.method == "curvelet"):
            decompress(args)

        else:
            raise ("method must be neural, wavelet, curvelet, or jpeg")
    else:
        raise ("mode must be compress or decompress")


if __name__ == "__main__":
    args = parse_args()
    main(args)
