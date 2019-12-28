# Attentive Exponentioanl Hawkes Process

This is a TensorFlow1.0 implementation of AEHN.

## Requirements
- python>=3.6
- tensorflow>=1.12.0

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Datasets

### Real world dataset
All datasets can be downloaded from [BaiduDisk](https://pan.baidu.com/s/1H6mfLB1MErHuh6gDrs88sw).

1. Retweet (retweet)
 - num of event: 3

2. StackOverflow (so)
 - num of event: 22

### Synthetic dataset
- [1d Exp Hawkes](https://pan.baidu.com/s/1IyummK-4ZbCsXjAPAQw6Ig)
- [2d Exp Hawkes](https://pan.baidu.com/s/1x75plmF_DYogY3IvN_gImQ)
- [3d Exp Hawkes](https://pan.baidu.com/s/1PgmZEY5ICFYXMpUKXj-k3Q)
- [5d Exp Hawkes](https://pan.baidu.com/s/1HX513dGqkk6EnrtaQSZdcQ)
- [10d Exp Hawkes](https://pan.baidu.com/s/1_Jc50wriOYsgMhcQ1QcR3g)

#### Loglike-per-event

|  |1d Hawkes |2d Hawkes  |  3d Hawkes |10d Hawkes | comment |
|--| --| ---|---|---| ---|
| NJSDE|  |    |   |  | epoch 500  |
| RMTPP|  | -2.246  | -1.398 |  | 200 epoch, 窗口50  |
| NHP |   -0.853|  -2.277  |  -1.377    |     | 200 epoch |
| AEHN | -0.690(80 epoch) |  -2.170 (30 epoch)   | -1.317 (50 epoch) |   | step=10 |


## Model Training
```bash
# Retweet
python train.py --config_filename=configs/retweet.yaml

# StackOverflow
python train.py --config_filename=configs/so.yaml
```

## Model Saving
The trained model is saved in a folder set up in configuration file.

## Model Evaluating
```bash
python eval.py --config_filename={saved_model_config_filename}
```
