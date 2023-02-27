# Data **Comp**ression for Distributed **A**cou**s**tic **S**ensing (Compass)

## Compression
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
python run.py --data-list1=test_decompressed.txt --data-path=../decompressed --dt=0.04 --maxlag=30  --mode=AM  --block-num1 1 --block-num2 2 --fixed-channels 300 500 700 900  --log-interval 1 --result-path results_decompressed
```

## Install Python Packages
```
pip install imageio, pillow
```

## Install PyCurvelab

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