# Attentive Exponentioanl Hawkes Process


## Results

### Retweet
| Model | method | type_acc | time_rmse |
| :----:| :-----: | :------: | :-------: |
|  RMTPP | 窗口20  | 49.80%   | 288.75   |
|  RMTPP | 窗口50  | 54.00%   | 309.92   |
| AEHN_simple |	hybrid|  59.91% | 274.64 |


### Stack Overflow (so)
| Model | method | type_acc | time_rmse |
| :----:| :-----: | :------: | :-------: |
|  RMTPP | 窗口20  | 43.10%   | 0.73   |
|  RMTPP | 窗口50  | 44.31%   | 0.71   |
| AEHN_simple |   hybrid  | 45.85%   |   0.64    |



## Loglikelihood-per-event
On simullar datasets:

| Model | Poisson | Hawkes-Exp | Hawkes-PL | Self-correlation | 2d Hawkes-Exp | 3d Hawkes-Exp|
| :--------: | :----------: | :--------: | :----------: | :--------: | :----------: | :--------: |
| Jump-Neural-SDE | -1.016 | -0.489 | -1.565 | -0.128 | xx | xx |
| Mei-NHP | -0.957 | -0.399 | -1.411 | -0.068| -2.287 | -1.377 |
| RMTPP | -1.018 | -0.553 | -1.518 | -0.102 | xx | xx|
| TF-NHP | -0.815 | -0.2499 | -1.176 | -0.006| xx | xx |
| AEHN_simple | xx | -0.656 | -1.616 | -0.086 | -2.135 | -1.255 |


### Simulated Hawkes data

- [1d Exp Hawkes](https://pan.baidu.com/s/1IyummK-4ZbCsXjAPAQw6Ig)
- [2d Exp Hawkes](https://pan.baidu.com/s/1x75plmF_DYogY3IvN_gImQ)
- [3d Exp Hawkes](https://pan.baidu.com/s/1PgmZEY5ICFYXMpUKXj-k3Q)
- [5d Exp Hawkes](https://pan.baidu.com/s/1HX513dGqkk6EnrtaQSZdcQ)
- [10d Exp Hawkes](https://pan.baidu.com/s/1YAGBwecVOkR_GC0mJ6NY3g)
- [20d Exp Hawkes](https://pan.baidu.com/s/1yPN9cVr23yCxbvE2XSanww)




|  |1d Hawkes |2d Hawkes  |  3d Hawkes |10d Hawkes| 10d Hawkes 0101 |20d Hawkes| comment |
|--|--|--|--|--|--|--|--|
| NJSDE|  -0.881|-2.324    |  -1.422 | -1.521 |  | | epoch 500  |
| RMTPP| -0.935 | -2.325  | -1.459 | -1.542 |   -1.114 | -1.388 | 500 epoch, 窗口50  |
| NHP |   -0.847|  -2.262  |  -1.373    | -1.487  | -1.057 | -1.180 | 500 epoch |
| AEHN | -0.690(80 epoch) |  -2.135   | -1.255 | -1.495 | -1.043(50 epoch)  | -1.157 | 1000 epoch, step=10 |

