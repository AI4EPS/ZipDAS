# %%
import os
import multiprocessing as mp
from zipdas.data import Config

# %%
def benchmarking(data_type, method, keep_ratio, quality):

    ### Noise
    if data_type == "noise":

        config = Config(
            data_path="noise_data",
            comp_path="results/compressed_noise",
            decomp_path="results/decompressed_noise",
            method=method,
            keep_ratio=keep_ratio,
            quality=quality,
            batch_nt=6000,
            plot_figure=False,
        )
        compress_cmd = f"python run.py --mode compress --batch_nt {config.batch_nt} --data_path {config.data_path} --data_format h5 --result_path {config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f} --method={config.method} --keep_ratio={config.keep_ratio} --quality={config.quality}"
        if config.plot_figure:
            compress_cmd += " --plot_figure"
        print(compress_cmd)
        os.system(compress_cmd)

        decompress_cmd = f"python run.py --mode decompress --batch_nt {config.batch_nt} --data_path {config.comp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method} --result_path {config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f} --method={config.method}"
        if config.plot_figure:
            decompress_cmd += " --plot_figure"
        print(decompress_cmd)
        os.system(decompress_cmd)

        cc_cmd = f"python CCTorch/run.py --data_list1=noise_data.txt --data_path={config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method}  --mode=AN  --block_size1 1 --block_size2 10 --fixed_channels 100 300 500 700 900  --dt=0.02 --maxlag=15 --result_path {config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method}/cctorch"
        print(cc_cmd)
        os.system(cc_cmd)

        plot_cmd = f"python CCTorch/scripts/plot_ambient_noise.py --result_path {config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method}/cctorch --figure_path {config.decomp_path}_{config.keep_ratio:.2f}_{config.quality:.0f}/{config.method}/cctorch  --fixed_channels 100 300 500 700 900"
        print(plot_cmd)
        os.system(plot_cmd)

# %%
if __name__ == "__main__":

    # %%
    keep_ratios = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 1.0]
    # keep_ratios = [1.0]
    # qualities = [1, 4, 10, 20, 40, 80]
    qualities = [1, 2, 4, 7, 10, 20, 40, 80]
    # qualities = [1.0]
    data_types = ["noise"]
    methods = ["jpeg", "wavelet", "curvelet", "neural"]
    methods = ["jpeg"]
    # keep_ratios = [0.01]
    # qualities = [1]
    ncpu = 1

    runs = []
    for method in methods:
        if method == "jpeg":
            keep_ratio = 1.0
            for quality in qualities:
                for data_type in data_types:
                    runs.append((data_type, method, keep_ratio, quality))
        else:
            quality = 1.0
            for keep_ratio in keep_ratios:
                for data_type in data_types:
                    runs.append((data_type, method, keep_ratio, quality))

    with mp.Manager() as manager:
        compression_rate = manager.dict()
        mse = manager.dict()
        ssim = manager.dict()
        psnr = manager.dict()
        corrcoef = manager.dict()
        with mp.get_context("spawn").Pool(ncpu) as p:
            p.starmap(benchmarking, runs)

    # ## save preprocessed data
    os.system("python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=jpeg --save_preprocess --quality=1.0 --batch_nt=600")

    # %%
    