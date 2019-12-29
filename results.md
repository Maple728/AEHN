# Attentive Exponentioanl Hawkes Process


## Results

### Retweet
| Model | method | type_acc | time_rmse |
| :----:| :-----: | :------: | :-------: |
| AEHN_simple |	hybrid|  59.91% | 274.64 |


### Stack Overflow (so)
| Model | method | type_acc | time_rmse |
| :----:| :-----: | :------: | :-------: |
| AEHN_simple |   hybrid  | 45.85%   |   0.64    |



## Loglikelihood-per-event
On simullar datasets:

| Model | Poisson | Hawkes-Exp | Hawkes-PL | Self-correlation | 2d Hawkes-Exp | 3d Hawkes-Exp
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

|  |1d Hawkes |2d Hawkes  |  3d Hawkes |10d Hawkes | comment |
|--| --| ---|---|---| ---|
| NJSDE|  |    |  -1.422 |  | epoch 500  |
| RMTPP| -0.961 | -2.246  | -1.398 | -1.550 | 200 epoch, 窗口50  |
| NHP |   -0.853|  -2.277  |  -1.377    | -1.505    | 200 epoch |
| AEHN | -0.690(80 epoch) |  -2.135   | -1.255 | -1.496 (90 epoch) | 1000 epoch, step=10 |
