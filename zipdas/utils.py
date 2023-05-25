# %%
import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlopen
from zipfile import ZipFile

# %%

def download_and_unzip(url, extract_to="./model"):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


# %%
def plot_result(args, filename, data):

    plt.clf()
    data = data.astype(np.float32)
    data -= np.mean(data)
    vmax = np.std(data) * 3
    vmin = -vmax
    plt.matshow(data, vmin=vmin, vmax=vmax, cmap="seismic")
    plt.title(f"[vmin, vmax] = [{vmin:.0f}, {vmax:.0f}]")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# %%