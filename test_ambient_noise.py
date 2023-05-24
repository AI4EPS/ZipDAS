# %%
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
if __name__ == "__main__":

    # %%
    # result_path = Path("../results")
    # figure_path = Path("figures")
    # result_path = Path("../results_decompressed_100")
    # figure_path = Path("figures_decompressed_100")

    result_path = Path("./results/raw_cc")
    figure_path = Path("./tests/figures")

    if not figure_path.exists():
        figure_path.mkdir(parents=True)

    tmp = []
    first_channels = [300, 500, 700, 900]
    with h5py.File(result_path / "AN_000_001.h5", "r") as fp:
        # pair_index = list(f.keys())
        # pair_index = pd.DataFrame(pair_index, columns=["pair_index"])
        # pair_index[["id1", "id2"]] = pair_index["pair_index"].apply(lambda x: pd.Series(x.split("_")))
        # pair_index["id1"] = pair_index["id1"].astype(int)
        # pair_index["id2"] = pair_index["id2"].astype(int)
        # pair_index.sort_values(by=["id1", "id2"], inplace=True)
        for chn in first_channels:
            # second_channels = sorted([int(x.split("/")[-1]) for x in fp[f"/{chn}"].keys()])
            data = []
            index = []
            print(fp[f"/{chn}"].keys())
            for c in sorted(fp[f"/{chn}"].keys(), key=lambda x: int(x.split("/")[-1])):
                data.append(fp[f"/{chn}/{c}"]["xcorr"][:])
                index.append(c)

        # for id1 in tqdm(pair_index["id1"].unique()):
        #     data = []
        #     for key in pair_index[pair_index["id1"] == id1]["pair_index"]:
        #         data.append(f[key][:])
        #     data = np.concatenate(data)

            data = np.stack(data)
            fig, axes = plt.subplots(1, 1)
            vmax = np.std(data)
            im = axes.imshow(data, vmin=-vmax, vmax=vmax, aspect="auto", cmap="RdBu")
            fig.colorbar(im, ax=axes)
            print(figure_path / f"result_{chn}.png")
            fig.savefig(figure_path / f"result_{chn}.png", dpi=300, bbox_inches="tight")


    method = "wavelet"
    # method = "curvelet"
    keep_ratios = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    for keep_ratio in keep_ratios:
        result_path = Path(f"./results/compressed_cc_{keep_ratio:.3f}/{method}")
        
        # result_path = Path("./results_decompressed_100")
        # figure_path = Path("./tests/figures_decompressed_100")

        # result_path = Path("../results_compressed_jpeg")
        # figure_path = Path("figures_compressed_jpeg")

        # result_path = Path("results_compressed_jpeg_500")
        # figure_path = Path("./tests/figures_compressed_jpeg_500")

        # result_path = Path("./results_decompressed")
        # figure_path = Path("./tests/figures_decompressed")

        if not figure_path.exists():
            figure_path.mkdir(parents=True)

        tmp = []
        first_channels = [300, 500, 700, 900]
        with h5py.File(result_path / "AN_000_001.h5", "r") as fp:
            # pair_index = list(f.keys())
            # pair_index = pd.DataFrame(pair_index, columns=["pair_index"])
            # pair_index[["id1", "id2"]] = pair_index["pair_index"].apply(lambda x: pd.Series(x.split("_")))
            # pair_index["id1"] = pair_index["id1"].astype(int)
            # pair_index["id2"] = pair_index["id2"].astype(int)
            # pair_index.sort_values(by=["id1", "id2"], inplace=True)
            for chn in first_channels:
                # second_channels = sorted([int(x.split("/")[-1]) for x in fp[f"/{chn}"].keys()])
                data = []
                index = []
                print(fp[f"/{chn}"].keys())
                for c in sorted(fp[f"/{chn}"].keys(), key=lambda x: int(x.split("/")[-1])):
                    data.append(fp[f"/{chn}/{c}"]["xcorr"][:])
                    index.append(c)

            # for id1 in tqdm(pair_index["id1"].unique()):
            #     data = []
            #     for key in pair_index[pair_index["id1"] == id1]["pair_index"]:
            #         data.append(f[key][:])
            #     data = np.concatenate(data)

                data = np.stack(data)
                fig, axes = plt.subplots(1, 1)
                vmax = np.std(data)
                im = axes.imshow(data, vmin=-vmax, vmax=vmax, aspect="auto", cmap="RdBu")
                fig.colorbar(im, ax=axes)
                print(figure_path / f"result_{chn}.png")
                fig.savefig(figure_path / f"result_{chn}_{keep_ratio:.3f}_{method}.png", dpi=300, bbox_inches="tight")

            # for i in range(1250):
            #     for j in [500]:
            #         pair_index = f"{i}_{j}"
            #         if pair_index in f:
            #             tmp.append(f[pair_index][:])
        # xcorr = np.array(tmp)
        # xcorr = xcorr.transpose(1, 0, 2)
        # print(xcorr.shape)
        # # raise

        # plt.figure()
        # # vmax = np.max(np.abs(xcorr[0, :, :]))
        # _, nch, nt = xcorr.shape
        # # xcorr[0, :, nt // 2 - 10 : nt // 2 + 11] *= 0.0
        # # vmax = np.std(xcorr[0, :, :])
        # mask = np.ones(nt)
        # mask[nt // 2 - 10 : nt // 2 + 11] = 0.0
        # # xcorr[0, :, nt // 2 - 10 : nt // 2 + 11] *= 0.0
        # vmax = np.std(xcorr[0, :, mask == 1.0]) * 3
        # plt.imshow(xcorr[0, :, :], cmap="seismic", vmax=vmax, vmin=-vmax)
        # plt.colorbar()
        # plt.savefig("xcorr.png", dpi=300)
        # ## TODO: cleanup writting

        # plt.figure()
        # ccall = xcorr[0, :, :]
        # max_lag = 30
        # vmax = np.percentile(np.abs(ccall), 99)
        # plt.imshow(
        #     # filter(ccall, 25, 1, 10),
        #     ccall,
        #     aspect="auto",
        #     vmax=vmax,
        #     vmin=-vmax,
        #     # extent=(-max_lag, max_lag, ccall.shape[0], 0),
        #     cmap="RdBu",
        # )
        # plt.colorbar()
        # plt.savefig("test_no_whitening_no_filtering.png", dpi=300)
        # plt.show()

    # %%