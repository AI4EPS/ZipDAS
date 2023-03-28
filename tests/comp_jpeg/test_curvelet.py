# %%
import pyct
import pyct.fdct2 as fdct2
import imageio.v3 as iio
import io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image
import pywt
from pathlib import Path
import h5py

# %%
# help(pyct.fdct2)

# %%
# img = iio.imread('imageio:astronaut.png')
# img = np.dot(img, [0.2989, 0.5870, 0.1140])
# img = img.astype(np.uint8)
# data = img
# print(f"img.shape: {img.shape}, img.dtype: {img.dtype}, img.min: {img.min()}, img.max: {img.max()}")

# %%
file_name = "data_raw/Ridgecrest_ODH3-2021-06-15 183838Z.h5"

with h5py.File(file_name, "r") as fp:
    data = fp["Data"][:, 2000:]
    

print(data.shape)
data = data[:, :10240]
# data = data[:, 1:] - data[:, :-1]
data = np.gradient(data, axis=-1)
# data -= np.mean(data, axis=-1, keepdims=True)
# data /= np.std(data, axis=-1, keepdims=True)

MAD = np.median(np.abs(data - np.median(data)))

# std = np.std(data[np.abs(data) < 6*np.std(data)])
# vmax = 0.1*np.std(data)
vmax = 10 * MAD
# vmax = 1.0
vmin = -vmax

fig, ax = plt.subplots(5, 1, squeeze=False, figsize=(20, 30), gridspec_kw={"height_ratios": [2, 2, 2, 1, 1]})
ax[0, 0].imshow(data, vmax=vmax, vmin=vmin, cmap="seismic", aspect='auto', interpolation="none")
ax[1, 0].imshow(data*(data > vmin)*(data < vmax), vmax=vmax, vmin=vmin, cmap="seismic", aspect='auto', interpolation="none")
ax[2, 0].imshow(data-data*(data > vmin)*(data < vmax), vmax=vmax, vmin=vmin, cmap="seismic", aspect='auto', interpolation="none")
ax[3, 0].hist(data.reshape(-1), bins=100)
ax[3, 0].set_yscale("log")
ax[4, 0].hist(data.reshape(-1), bins=100, range=(vmin, vmax))
ax[4, 0].set_yscale("log")
plt.savefig("test_data_raw_stats.png", dpi=300, bbox_inches="tight")


# # %%
# file_name = "data_raw/Ridgecrest_ODH3-2021-06-15 183838Z.h5"

# with h5py.File(file_name, "r") as fp:
#     data = fp["Data"][:, 2000:]


# %%
print(data.shape)
data = data[:, :10240]
data = np.gradient(data, axis=-1)
# data -= np.mean(data, axis=-1, keepdims=True)
# data /= np.std(data, axis=-1, keepdims=True)
# vmax = 5*np.std(data)
# vmin = -vmax

# data -= np.mean(data, axis=-1, keepdims=True)
# data /= np.std(data, axis=-1, keepdims=True)

MAD = np.median(np.abs(data - np.median(data)))
vmax = 10 * MAD
vmin = -vmax

fig, ax = plt.subplots(5, 1, squeeze=False, figsize=(20, 30), gridspec_kw={"height_ratios": [2, 2, 2, 1, 1]})
ax[0, 0].imshow(data, vmax=vmax/2, vmin=vmin/2, cmap="seismic", aspect='auto', interpolation="none")
ax[1, 0].imshow(data*(data > vmin)*(data < vmax), vmax=vmax/2, vmin=vmin/2, cmap="seismic", aspect='auto', interpolation="none")
ax[2, 0].imshow(data-data*(data > vmin)*(data < vmax), vmax=vmax/2, vmin=vmin/2, cmap="seismic", aspect='auto', interpolation="none")
ax[3, 0].hist(data.reshape(-1), bins=100)
ax[3, 0].set_yscale("log")
ax[4, 0].hist(data.reshape(-1), bins=100, range=(vmin, vmax))
ax[4, 0].set_yscale("log")
plt.savefig("test_data_raw_norm_time_stats.png", dpi=300, bbox_inches="tight")

# %%
file_name = "data/Ridgecrest_ODH3-2021-06-15 183838Z.h5"

with h5py.File(file_name, "r") as fp:
    data = fp["Data"][:, 2000:]

# %%
print(data.shape)
data = data[:, :10240]
# data = data[:, 1:] - data[:, :-1]
# data -= np.mean(data, axis=-1, keepdims=True)
# data /= np.std(data, axis=-1, keepdims=True)
vmax = 5*np.std(data)
vmin = -vmax
np.savez("test_data.npz", data=data.astype(np.float32))

fig, ax = plt.subplots(5, 1, squeeze=False, figsize=(20, 30), gridspec_kw={"height_ratios": [2, 2, 2, 1, 1]})
ax[0, 0].imshow(data, vmax=vmax/2, vmin=vmin/2, cmap="seismic", aspect='auto', interpolation="none")
ax[1, 0].imshow(data*(data > vmin)*(data < vmax), vmax=vmax/2, vmin=vmin/2, cmap="seismic", aspect='auto', interpolation="none")
ax[2, 0].imshow(data-data*(data > vmin)*(data < vmax), vmax=vmax/2, vmin=vmin/2, cmap="seismic", aspect='auto', interpolation="none")
ax[3, 0].hist(data.reshape(-1), bins=100)
ax[3, 0].set_yscale("log")
ax[4, 0].hist(data.reshape(-1), bins=100, range=(vmin*2, vmax*2))
ax[4, 0].set_yscale("log")
plt.savefig("test_data_stats.png", dpi=300, bbox_inches="tight")

# %%
data_unit8 = ((data - vmin) / (vmax - vmin) * 255)
data_unit8 = data_unit8.astype(np.uint8)

plt.figure()
plt.imshow(data, cmap='seismic', vmin=vmin/2, vmax=vmax/2, aspect='auto' , interpolation="none")
plt.savefig('test_data.png', dpi=300,  bbox_inches="tight")
# iio.imwrite("test_data.jpg", data_unit8, plugin="pillow", extension=".jpg", quality=5)
iio.imwrite("test_data.jpg", data_unit8, plugin="pillow", extension=".jpg", quality=10)
tmp = iio.imread("test_data.jpg")
tmp = (tmp / 255.0 * (vmax - vmin) + vmin)
plt.figure()
plt.imshow(tmp, cmap='seismic', vmin=vmin/2, vmax=vmax/2, aspect='auto', interpolation="none")
plt.savefig('test_data.jpg.png', dpi=300, bbox_inches="tight")
# iio.imwrite("test_data.j2k", data_unit8, plugin="pillow", extension=".j2k", quality_mode="rates", quality_layers=[50], irreversible=True)
iio.imwrite("test_data.j2k", data_unit8, plugin="pillow", extension=".j2k", quality_mode="rates", quality_layers=[20], irreversible=True)
tmp = iio.imread("test_data.j2k")
tmp = (tmp / 255.0 * (vmax - vmin) + vmin)
plt.figure()
plt.imshow(tmp, cmap='seismic', vmin=vmin/2, vmax=vmax/2, aspect='auto', interpolation="none")
plt.savefig('test_data.j2k.png', dpi=600, bbox_inches="tight")

# %%
nx, nt = data.shape
ct_config = {
'n': (nx, nt),
'nbs': 3,
'nba': 1,
'ac': False,
}
print(data.shape)
print(data.dtype)
print(data.min(), data.max())
func_fdct2 = fdct2(ct_config["n"], ct_config["nbs"], ct_config["nba"], ct_config["ac"], norm=False, vec=True, cpx=False)
data_ct = func_fdct2.fwd(data)
ct_sort = np.sort(np.abs(data_ct.reshape(-1)))

# %%
wt_config = {
'n': 4,
"w": "db1",
}
coeffs = pywt.wavedec2(data, wavelet=wt_config["w"], level=wt_config["n"])
data_wt, coeff_slices = pywt.coeffs_to_array(coeffs)
wt_sort = np.sort(np.abs(data_wt.reshape(-1)))

keep = 0.1

# %%
# keep = 1.0
threshold = wt_sort[int(len(wt_sort) * (1-keep))]
data_wt_filt = data_wt
data_wt_filt[data_wt_filt < threshold] = 0.0
min_wt = data_wt_filt.min()
max_wt = data_wt_filt.max()
print(f"min_wt: {min_wt}, max_wt: {max_wt}")
data_wt_quant  = ((data_wt_filt - min_wt) / (max_wt - min_wt) * 255.0).astype(np.uint8) 
# data_wt_quant  = data_wt_filt
np.savez_compressed("test_data_wt.npz", data=data_wt_quant, allow_pickle=False)

data_wt = np.load("test_data_wt.npz")["data"].astype(np.float32)
# data_wt = data_wt_quant
data_wt = (data_wt / 255.0) * (max_wt - min_wt) + min_wt
data_wt = pywt.array_to_coeffs(data_wt, coeff_slices, output_format='wavedec2')
data_wt = pywt.waverec2(data_wt, wavelet=wt_config["w"])
# matplotlib.image.imsave('test_data_wt.npz.png', data_wt, cmap='gray')
plt.figure()
plt.imshow(data_wt, cmap='seismic', vmin=vmin/2, vmax=vmax/2, aspect='auto', interpolation="none")
plt.savefig('test_data_wt.npz.png', dpi=300, bbox_inches="tight")

# %%
keep = keep/2.0
threshold = ct_sort[int(len(ct_sort) * (1-keep))]
data_ct_filt = data_ct
data_ct_filt[np.abs(data_ct_filt) < threshold] = 0.0
min_ct = data_ct_filt.min()
max_ct = data_ct_filt.max()
print(f"min_ct: {min_ct}, max_ct: {max_ct}")
data_ct_quant  = ((data_ct_filt - min_ct) / (max_ct - min_ct) * 255.0).astype(np.uint8)
# data_ct_quant = data_ct_filt 
np.savez_compressed("test_data_ct.npz", data=data_ct_quant, allow_pickle=False)

data_ct = np.load("test_data_ct.npz")["data"].astype(np.float32)
# data_ct = data_ct_quant
data_ct = (data_ct / 255.0) * (max_ct - min_ct) + min_ct
data_ct = func_fdct2.inv(data_ct)
# matplotlib.image.imsave('test_data_ct.npz.png', data_ct, cmap='gray')
plt.figure()
plt.imshow(data_ct, cmap='seismic', vmin=vmin/2, vmax=vmax/2, aspect='auto', interpolation="none")
plt.savefig('test_data_ct.npz.png', dpi=300, bbox_inches="tight")

# %%
# min_ct = 200
cut_off = 600
# min_ct = data_ct.min()
# max_ct = data_ct.max()
# print(f"{min_ct = }, {max_ct = }")
data_ct_comp = data_ct.copy()
data_ct_comp[np.abs(data_ct_comp) < cut_off] = 0.0
# data_ct_comp[np.abs(data_ct_comp) > cut_off] = 0.0
min_ct = data_ct_comp.min()
max_ct = data_ct_comp.max()

# data_ct_comp  = (data_ct_comp / max_ct * 256.0).astype(np.uint8) 
data_ct_comp  = ((data_ct_comp - min_ct) / (max_ct - min_ct) * 256.0).astype(np.uint8) 
print(f"{data_ct_comp.shape = }, {data_ct_comp.dtype = }")

# %%
np.savez_compressed("test_data.npz", data=data_ct_comp, allow_pickle=False)

# data_ct_comp = np.load("test_data.npz")['data']
# data_ct_comp = data_ct_comp.astype(np.float32) * max_ct / 256.0
data_ct_comp = data_ct_comp.astype(np.float32) * (max_ct - min_ct) / 256.0 + min_ct

# %%
data_ = func_fdct2.inv(data_ct)
data_comp = func_fdct2.inv(data_ct_comp)

# im = iio.
# plt.figure()
# plt.imshow(data_comp, cmap='gray')
# plt.savefig('test_data.npz.png')
matplotlib.image.imsave('test_data.npz.png', data_comp, cmap='gray')

# %%
print(f"{data_ct_comp.shape = }")
print(f"{data_.shape = }")
print(f"{data_comp.shape = }")


# %%
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(10, 5), sharex=True, sharey=True)
ax[0,0].hist(data_ct, bins=100, range=(data_ct.min(), data_ct.max()))
ax[0,0].set_yscale('log')
ax[1,0].hist(data_ct_comp, bins=100)
ax[1,0].set_yscale('log')
plt.savefig('test_curvelet_hist.png')
plt.show()

# %%

fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(10, 5))
ax[0,0].imshow(data, cmap='gray')
ax[0,1].imshow(data_, cmap='gray')
ax[0,2].imshow(data_comp, cmap='gray')
plt.savefig('test_curvelet.png')
plt.show()



