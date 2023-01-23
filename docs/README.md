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