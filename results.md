# Attentive Exponentioanl Hawkes Process


## Results

### Retweet
| Model | type_acc | time_rmse |
| :----:| :------: | :-------: |
|  RMTPP | 54.14%   | 288.73 |
|  NHP | 54.97%   | 276.64   |
|  jump | 54.08%   | 306.32 |
| AEHN | 60.21% | 255.75 |


### Stack Overflow (so)
| Model | type_acc | time_rmse |
| :----:|  :------: | :-------: |
|  RMTPP |  45.18%   | 0.73   |
|  NHP | 45.50%   | 0.71   |
|  jump | 45.55%   | 0.82 |
| AEHN |  45.55%   | 0.66  |

### Bund Future 
| Model | type_acc | time_rmse |
| :----:|:------: | :-------: |
|  RMTPP | 63.46%   | 32.32   |
|  NHP | xx%   | xx  |
|  jump | 60.24%   | 40.66 |
| AEHN | 63.83%   |   5.66   |


## Loglikelihood-per-event
On synthetic datasets:

| Model | Poisson | Hawkes-Exp | Hawkes-PL | Self-correlation |
| :--------: | :----------: | :--------: | :----------: | :--------: |
| Jump-Neural-SDE | -1.016 | -0.489 | -1.565 | -0.128 |
| Mei-NHP | -0.946 | -0.399 | -1.411 | -0.068|
| RMTPP | -1.018 | -0.553 | -1.518 | -0.102 |
| TF-NHP | -0.815 | -0.2499 | -1.176 | -0.006|
| AEHN_simple | -0.878 | -0.656 | -1.616 | -0.056 |


### Simulated Hawkes data

- [1d Exp Hawkes](https://pan.baidu.com/s/1IyummK-4ZbCsXjAPAQw6Ig)
- [2d Exp Hawkes](https://pan.baidu.com/s/1x75plmF_DYogY3IvN_gImQ)
- [3d Exp Hawkes](https://pan.baidu.com/s/1PgmZEY5ICFYXMpUKXj-k3Q)
- [5d Exp Hawkes](https://pan.baidu.com/s/1HX513dGqkk6EnrtaQSZdcQ)
- [10d Exp Hawkes](https://pan.baidu.com/s/1YAGBwecVOkR_GC0mJ6NY3g)
- [20d Exp Hawkes](https://pan.baidu.com/s/1yPN9cVr23yCxbvE2XSanww)




|  |1d Hawkes |2d Hawkes  |  3d Hawkes |10d Hawkes| 10d Hawkes 0101 |20d Hawkes| comment |
|--|--|--|--|--|--|--|--|
| NJSDE| -0.880 | -2.322 |  -1.409 |   | -1.058 | -1.168| epoch 500  |
| NJSDE|  -0.881|-2.324    |  -1.422 | -1.521 |  | | epoch 200  |
| RMTPP| -0.935 | -2.325  | -1.459 | -1.542 |   -1.114 | -1.388 | 500 epoch, 窗口50  |
| NHP |   -0.847|  -2.262  |  -1.373    | -1.487  | -1.057 | -1.180 | 500 epoch |
| AEHN | -0.656 |  -2.135   | -1.255 | -1.495 | -1.043 | -1.157 | 1000 epoch, step=10 |

