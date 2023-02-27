# %%
import pyct
import pyct.fdct2 as fdct2
import imageio.v3 as iio
import io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image
import pywt

# %%
# help(pyct.fdct2)

# %%
img = iio.imread('imageio:astronaut.png')
img = np.dot(img, [0.2989, 0.5870, 0.1140])
img = img.astype(np.uint8)
data = img
print(f"img.shape: {img.shape}, img.dtype: {img.dtype}, img.min: {img.min()}, img.max: {img.max()}")

# %%
iio.imwrite("test_data.png", img, format="png")
iio.imwrite("test_data.jpg", img, plugin="pillow", extension=".jpg", quality=5)
tmp = iio.imread("test_data.jpg")
iio.imwrite("test_data.jpg.png", tmp, format="png")
iio.imwrite("test_data.j2k", img, plugin="pillow", extension=".j2k", quality_mode="rates", quality_layers=[50], irreversible=True)
tmp = iio.imread("test_data.j2k")
iio.imwrite("test_data.j2k.png", tmp, format="png")


# %%
ct_config = {
'n': (512, 512),
'nbs': 3,
'nba': 1,
'ac': False,
}
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

keep = 1.0

# %%
# keep = 1.0
threshold = wt_sort[int(len(wt_sort) * (1-keep))]
data_wt_filt = data_wt
# data_wt_filt[data_wt_filt < threshold] = 0.0
min_wt = data_wt_filt.min()
max_wt = data_wt_filt.max()
print(f"min_wt: {min_wt}, max_wt: {max_wt}")
# data_wt_quant  = ((data_wt_filt - min_wt) / (max_wt - min_wt) * 255.0).astype(np.uint8) 
data_wt_quant  = data_wt_filt
np.savez_compressed("test_data_wt.npz", data=data_wt_quant, allow_pickle=False)

# data_wt = np.load("test_data_wt.npz")["data"].astype(np.float32)
data_wt = data_wt_quant
# data_wt = (data_wt_filt / 255.0) * (max_wt - min_wt) + min_wt
data_wt = pywt.array_to_coeffs(data_wt, coeff_slices, output_format='wavedec2')
data_wt = pywt.waverec2(data_wt, wavelet=wt_config["w"])
matplotlib.image.imsave('test_data_wt.npz.png', data_wt, cmap='gray')

# %%
# keep = 1.0
threshold = ct_sort[int(len(ct_sort) * (1-keep))]
data_ct_filt = data_ct
# data_ct_filt[np.abs(data_ct_filt) < threshold] = 0.0
min_ct = data_ct_filt.min()
max_ct = data_ct_filt.max()
print(f"min_ct: {min_ct}, max_ct: {max_ct}")
# data_ct_quant  = ((data_ct_filt - min_ct) / (max_ct - min_ct) * 255.0).astype(np.uint8)
data_ct_quant = data_ct_filt 
np.savez_compressed("test_data_ct.npz", data=data_ct_quant, allow_pickle=False)

# data_ct = np.load("test_data_ct.npz")["data"].astype(np.float32)
data_ct = data_ct_quant
# data_ct = (data_ct / 255.0) * (max_ct - min_ct) + min_ct
data_ct = func_fdct2.inv(data_ct)
matplotlib.image.imsave('test_data_ct.npz.png', data_ct, cmap='gray')


raise
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



