# ZipDAS: Distributed Acoustic Sensing Data Compression


## Installation

```
pip install -r requirements.txt
```

## Usage

### Compression using Wavelet (JPEG2000)

*Compression*
```
python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=jpeg --save_preprocess --quality=80 --plot_figure
```
*Decompression:*
```
python run.py --mode decompress --data_path results/compressed_noise/jpeg --result_path results/decompressed_noise --method=jpeg --plot_figure
```

### Compression using Neural Network models

To use GPU acceleration, please follow this instruction: https://www.tensorflow.org/install/pip

*Compression and decompression:*
```
python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=neural  --model_path=model --batch_nt=10240  --plot_figure
```
```
python run.py --mode decompress --data_path results/compressed_noise/neural --result_path results/neural/decompressed_noise --method=neural --model_path model --plot_figure
```

*Training:*
```
python train.py  --model_path model  --data_path noise_data --result_path training --format=h5
```
*Test:*
```
python train.py --mode test  --model_path model  --data_path tests/data --result_path tmp --format=h5 --batch 0 --nt 2048
```

### Optional: Compression using Curvelet

```
python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=curvelet --keep_ratio 0.01 --batch_nt 6000 --plot_figure 
```
```
python run.py --mode decompress --data_path results/compressed_noise/curvelet --result_path results/decompressed_noise --method=curvelet --plot_figure
```

<!-- ```
python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=wavelet --keep_ratio 0.01 --batch_nt 6000 --plot_figure 
```
```
python run.py --mode decompress --data_path results/compressed_noise/wavelet --result_path results/decompressed_noise --method=wavelet --plot_figure
``` -->

## Application: Ambient Noise

*Raw data:*
```
python CCTorch/run.py --data_list1=noise_data.txt --data_path=noise_data  --mode=AN  --block_size1 1 --block_size2 100 --fixed_channels 100 300 500 700 900  --dt=0.02 --maxlag=15 --temporal_gradient --result_path results/cctorch_noise
```
```
python CCTorch/tests/test_ambient_noise.py --result_path results/cctorch_noise --figure_path figures/cctorch_noise  --fixed_channels 100 300 500 700 900
```

<!-- *Preprocess data*
```
python CCTorch/run.py --data_list1=noise_data.txt --data_path=results/compressed_noise/jpeg/preprocess  --mode=AN  --block_size1 1 --block_size2 100 --fixed_channels 100 300 500 700 900  --dt=0.02 --maxlag=15  --result_path results/cctorch_preprocess_noise
```
```
python CCTorch/tests/test_ambient_noise.py --result_path results/cctorch_preprocess_noise --figure_path results/cctorch_preprocess_noise  --fixed_channels 100 300 500 700 900
``` -->

*Compressed data:*
```
python CCTorch/run.py --data_list1=noise_data.txt --data_path=results/decompressed_noise/jpeg  --mode=AN  --block_size1 1 --block_size2 100 --fixed_channels 100 300 500 700 900  --dt=0.02 --maxlag=15  --result_path results/cctorch_decompressed_noise
```
```
python CCTorch/tests/test_ambient_noise.py --result_path results/cctorch_decompressed_noise --figure_path figures/cctorch_decompressed_noise  --fixed_channels 100 300 500 700 900
```

## Optional: Install PyCurvelab for curvelet compression

[FFTW 2.1.5](https://www.fftw.org/)

```
wget http://www.fftw.org/fftw-2.1.5.tar.gz
tar -xvf fftw-2.1.5.tar.gz
cd fftw-2.1.5
./configure --with-pic --prefix=/home/weiqiang/.local/
make
make install
```

[CurveLab](http://www.curvelet.org/)
```
tar -xvf CurveLab-2.1.3.tar.gz
cd CurveLab-2.1.3
vi makefile.opt
make
make install
```

[SWIG](https://www.swig.org/)
```
tar -xvf swig-4.1.1.tar.gz
cd swig-4.1.1
./configure --prefix=/home/weiqiang/.local/
make
make install
```

[PyCurvelab](https://github.com/slimgroup/PyCurvelab)
```
git clone https://github.com/slimgroup/PyCurvelab.git
cd PyCurvelab
export FDCT=/home/weiqiang/Research/DASCompression/tests/comp_curvelet/CurveLab-2.1.3/
export FFTW=/home/weiqiang/.local/
python setup.py build install
```


<!-- python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=jpeg --save_preprocess --quality=80 --batch_nt 6000 --plot_figure
python run.py --mode decompress --data_path results/compressed_noise/jpeg --result_path results/decompressed_noise --method=jpeg --plot_figure

python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=wavelet --keep_ratio 0.01 --batch_nt 6000 --plot_figure 
python run.py --mode decompress --data_path results/compressed_noise/wavelet --result_path results/decompressed_noise --method=wavelet --plot_figure

python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=curvelet --keep_ratio 0.01 --batch_nt 6000 --plot_figure 
python run.py --mode decompress --data_path results/compressed_noise/curvelet --result_path results/decompressed_noise --method=curvelet --plot_figure


python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=wavelet --keep_ratio 1.0 --batch_nt 6000 --plot_figure 
python run.py --mode decompress --data_path results/compressed_noise/wavelet --result_path results/decompressed_noise --method=wavelet --plot_figure

python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=curvelet --keep_ratio 1.0 --batch_nt 6000 --plot_figure 
python run.py --mode decompress --data_path results/compressed_noise/curvelet --result_path results/decompressed_noise --method=curvelet --plot_figure


python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise/jpeg --method=jpeg --quality=50

python run.py --mode decompress --data_path results/compressed_noise/jpeg --result_path results/decompressed_noise/jpeg --method=jpeg

python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=jpeg --quality=24 --batch_nt=12000 && python run.py --mode decompress --data_path results/compressed_noise/jpeg --result_path results/decompressed_noise --method=jpeg  --plot_figure

python run.py --mode compress --data_path noise_data --data_format h5 --result_path results/compressed_noise --method=jpeg --quality=24 --plot_figure --batch_nt=12000 && python run.py --mode decompress --data_path results/compressed_noise/jpeg --result_path results/decompressed_noise --method=jpeg  --plot_figure

python run.py --mode compress --data_path event_data --data_format h5 --result_path results/compressed_event --method=jpeg --quality=10 && python run.py --mode decompress --data_path results/compressed_event/jpeg --result_path results/decompressed_event --method=jpeg  --plot_figure -->

<!-- ## Compression
```
python run.py --model_path model --mode compress --data_path tests/data --format h5 --result_path compressed --plot_figure
```

## Decompression
```
python run.py --model_path model --mode decompress --data_path compressed --result_path decompressed --plot_figure
```

## Training:
```
python train.py  --model_path model  --data_path tests/data --result_path training --format=h5
```

## Test:
```
python train.py --mode test  --model_path model  --data_path tests/data --result_path tmp --format=h5 --batch 0 --nt 2048
```

## Run CCTorch:
```
python run.py --data-list1=test_decompressed.txt --data-path=../decompressed --dt=0.04 --maxlag=30  --mode=AN  --block-size1 1300 --block-size2 1300 --fixed-channels 300 500 700 900  --log-interval 1 --result-path results_decompressed
```

## Install Python Packages
```
pip install imageio, pillow
``` -->


<!-- ## Experiments

## Ambient Noise

### compress data
```
python run.py --mode compress --data_path CCTorch/event_data/ --format h5 --result_path compressed_template --method=wavelet --keep_ratio=0.1 --plot_figure
```

### decompress data
```
python run.py --mode decompress --data_path compressed_template/wavelet --result_path decompressed_template/ --method=wavelet --plot_figure
```

### run CCTorch
#### RAW 
```
python ../CCTorch/run.py --pair-list=templates_raw/event_pair.txt  --data-path=templates_raw/template.dat --data-format=memmap --config=templates_raw/config.json  --batch-size=512  --result-path=templates_raw/ccpairs```
### COMPRESSED
```
```
python ../CCTorch/run.py --pair-list=templates_compressed/event_pair.txt  --data-path=templates_compressed/template.dat --data-format=memmap --config=templates_compressed/config.json  --batch-size=512  --result-path=templates_compressed/ccpairs
```



## Earthquake
### convert data
```
python CCTorch/tests/convert_templates.py
```

### compress data
```
python run.py --mode compress --data_path CCTorch/event_data/ --format h5 --result_path compressed_template --method=wavelet --keep_ratio=0.1 --plot_figure
```

### decompress data
```
python run.py --mode decompress --data_path compressed_template/wavelet --result_path decompressed_template/ --method=wavelet --plot_figure
```

### cut templates
```
python CCTorch/tests/convert_templates.py
```

### run CCTorch
#### RAW 
```
python ../CCTorch/run.py --pair-list=templates_raw/event_pair.txt  --data-path=templates_raw/template.dat --data-format=memmap --config=templates_raw/config.json  --batch-size=512  --result-path=templates_raw/ccpairs
```
### COMPRESSED
```
python ../CCTorch/run.py --pair-list=templates_compressed/event_pair.txt  --data-path=templates_compressed/template.dat --data-format=memmap --config=templates_compressed/config.json  --batch-size=512  --result-path=templates_compressed/ccpairs
``` -->

