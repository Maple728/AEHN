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
All datasets can be downloaded from [BaiduDisk](https://pan.baidu.com/s/1H6mfLB1MErHuh6gDrs88sw).


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
# Retweet
python eval.py --config_filename={saved_model_config_filename}
```

## Results

| Model | Dataset | type_acc | time_rmse |
| :----:| :-----: | :------: | :-------: |
| AEHN_simple |   so  | 45.83%   |   0.64    |
| AEHN_simple |retweet|  59.91% | 274.64 |