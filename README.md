## Train
```angular2html
CUDA_VISIBLE_DEVICES=[GPU_ID] python3 train.py single --net p3d199 --tag [TAG] -tf -rs --modality MODALITY [-rd] [-m] [-dl]
# example
CUDA_VISIBLE_DEVICES=0 python3 train.py single --net p3d199 --tag test -tf -rs --modality RGB
```

## Test
```angular2html
CUDA_VISIBLE_DEVICES=[GPU_ID] python3 test.py single --net p3d199 --modality MODALITY [--resume CHECKPOINT] [--test_set valid] [--ensemble] [-m]
# exmaple
CUDA_VISIBLE_DEVICES=0 python3 test.py single --net i3d --modality RGB

```
