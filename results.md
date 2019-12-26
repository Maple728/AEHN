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

| Model | Poisson | Hawkes-Exp | Hawkes-PL | Self-correlation |
| :--------: | :----------: | :--------: | :----------: | :--------: |
| Jump-Neural-SDE | -1.016 | -0.489 | -1.565 | -0.128 |
| Mei-NHP | -0.957 | -0.399 | -1.411 | -0.068|
| RMTPP | -1.018 | -0.553 | -1.518 | -0.102 |
| TF-NHP | -0.815 | -0.2499 | -1.176 | -0.006|
| AEHN_simple | xx | -0.656 | -1.616 | -0.086 |

