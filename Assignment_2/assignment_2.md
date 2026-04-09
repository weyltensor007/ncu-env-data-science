# Assignment_2

**Student Name:** 郭忠侑

## 1. Complete Exercise 6.5 in Hsieh’s book. Please build an MLP NN model and use the cross-validation technique to tune at least one model hyperparameter other than the learning rate.

![Problem1](imgs/6.5.jpg)

I've chosen the ==number of hidden neurons==(n_hidden) as the hyperparameter of my ELM. Also, I did the 5-fold cross validation to select the best n_hidden as 5.
Here's my output result of cross validation(each item in CV rmse results means n_hidden vs. corresponding rmse):

```python
CV rmse results: {5: 490.816, 10: 526.899, 20: 1781.641, 50: 34675.982, 100: 15026.577}
Best n_hidden: 5
```

Finally, let's compare the rmse and the correlation coefficient between the ELM and MLR:

```python
rmse_ELM_ensembles=492.876, rmse_MLR=519.302
corr_ELM_ensembles=0.493, corr_MLR=0.37
```

It's easy to observe that rmse is lower and corr is higher for ELM, indicating it's a better model than the LMR.

[Problem1 Code](https://github.com/weyltensor007/ncu-env-data-science/blob/main/Assignment_2/problem1.py)

<div class="page"></div>

## 2. Complete Exercise 8.1 in Hsieh’s book. Please tune the learning rate for the MLP NN model.

![Problem2](imgs/8.1.jpg)

I used `Keras` to build and train NN models, in particular, using the `callbacks.EarlyStopping` utility helped the tuning of the learning rate. Also, the training of NN models take time, so I split the workflow into 2 steps, namely train/pred and evaluation by saving the predicting values as `npy` file. Finally here is my full resulting table:


|      | sigma          | n_hidden | n_ensemble | avg_rmse | ensemble_rmse |     best_lr |
| ---: | :------------- | -------: | ---------: | -------: | ------------: | ----------: |
|    0 | noise-0.5sigma |        2 |        100 |  28.7708 |       25.1531 | 1.61026e-05 |
|    1 | noise-0.5sigma |        2 |         25 |  28.7547 |       24.9185 | 0.000259294 |
|    2 | noise-0.5sigma |        2 |         50 |   29.274 |       24.4173 | 0.000259294 |
|    3 | noise-0.5sigma |        3 |        100 |  23.6217 |       23.4006 |         0.1 |
|    4 | noise-0.5sigma |        3 |         25 |   29.141 |       25.6851 | 3.56225e-05 |
|    5 | noise-0.5sigma |        3 |         50 |  27.0627 |       24.0133 | 5.29832e-05 |
|    6 | noise-0.5sigma |        4 |        100 |  24.8545 |       24.0693 |  0.00621017 |
|    7 | noise-0.5sigma |        4 |         25 |  25.3489 |         23.76 | 2.39503e-05 |
|    8 | noise-0.5sigma |        4 |         50 |  24.1676 |        23.769 |  0.00923671 |
|    9 | noise-0.5sigma |        5 |        100 |  27.8978 |       24.9391 |  0.00011721 |
|   10 | noise-0.5sigma |        5 |         25 |  27.1655 |       25.4327 | 7.27895e-06 |
|   11 | noise-0.5sigma |        5 |         50 |  23.7184 |       23.5349 |   0.0452035 |
|   12 | noise-0.5sigma |        6 |        100 |   27.263 |       24.7176 |       1e-06 |
|   13 | noise-0.5sigma |        6 |         25 |  26.8245 |        24.384 | 2.39503e-05 |
|   14 | noise-0.5sigma |        6 |         50 |  23.7208 |        23.509 |   0.0672336 |
|   15 | noise-0.5sigma |        7 |        100 |  24.6896 |        23.865 |  0.00280722 |
|   16 | noise-0.5sigma |        7 |         25 |  27.6918 |       26.3387 | 3.56225e-05 |
|   17 | noise-0.5sigma |        7 |         50 |  26.3278 |       24.6511 | 0.000259294 |
|   18 | noise-0.5sigma |        8 |        100 |  23.6707 |       23.5122 |         0.1 |
|   19 | noise-0.5sigma |        8 |         25 |  25.8088 |       24.3035 | 1.48735e-06 |
|   20 | noise-0.5sigma |        8 |         50 |  23.6657 |       23.5672 |   0.0137382 |
|   21 | noise-1sigma   |        2 |        100 |  31.6434 |       31.1158 |   0.0137382 |
|   22 | noise-1sigma   |        2 |         25 |  34.7381 |       31.3533 | 0.000174333 |
|   23 | noise-1sigma   |        2 |         50 |  32.8571 |       31.1651 |  0.00280722 |
|   24 | noise-1sigma   |        3 |        100 |  34.5309 |       31.6927 | 1.48735e-06 |
|   25 | noise-1sigma   |        3 |         25 |  35.6238 |       33.6718 | 0.000385662 |
|   26 | noise-1sigma   |        3 |         50 |  33.7594 |       31.9793 | 2.39503e-05 |
|   27 | noise-1sigma   |        4 |        100 |  33.0722 |       31.4156 |  0.00188739 |
|   28 | noise-1sigma   |        4 |         25 |  33.5855 |       31.2513 | 3.29034e-06 |
|   29 | noise-1sigma   |        4 |         50 |    31.04 |       30.8593 |         0.1 |
|   30 | noise-1sigma   |        5 |        100 |  33.5398 |       31.5256 | 3.56225e-05 |
|   31 | noise-1sigma   |        5 |         25 |  33.0377 |       31.9842 |  0.00280722 |
|   32 | noise-1sigma   |        5 |         50 |   32.747 |       31.4035 | 1.08264e-05 |
|   33 | noise-1sigma   |        6 |        100 |  31.1527 |       30.8965 |         0.1 |
|   34 | noise-1sigma   |        6 |         25 |  32.5912 |       31.7305 |  0.00126896 |
|   35 | noise-1sigma   |        6 |         50 |  32.3214 |       31.3423 |  0.00188739 |
|   36 | noise-1sigma   |        7 |        100 |  34.0679 |       32.2407 | 1.08264e-05 |
|   37 | noise-1sigma   |        7 |         25 |  30.9552 |       30.8373 |         0.1 |
|   38 | noise-1sigma   |        7 |         50 |  30.9456 |       30.8357 |    0.030392 |
|   39 | noise-1sigma   |        8 |        100 |  33.2887 |       31.9374 | 7.88046e-05 |
|   40 | noise-1sigma   |        8 |         25 |  32.1405 |       31.5246 |  0.00126896 |
|   41 | noise-1sigma   |        8 |         50 |  31.0152 |        30.878 |   0.0672336 |
|   42 | noise-2sigma   |        2 |        100 |   51.371 |       49.8441 | 2.21222e-06 |
|   43 | noise-2sigma   |        2 |         25 |  52.2632 |       50.0285 |  0.00126896 |
|   44 | noise-2sigma   |        2 |         50 |  49.6787 |        49.564 |   0.0452035 |
|   45 | noise-2sigma   |        3 |        100 |  51.5901 |       50.1737 |  4.8939e-06 |
|   46 | noise-2sigma   |        3 |         25 |  51.3415 |       50.2732 | 1.61026e-05 |
|   47 | noise-2sigma   |        3 |         50 |  50.5423 |       50.1583 |  0.00417532 |
|   48 | noise-2sigma   |        4 |        100 |  49.6666 |       49.5483 |         0.1 |
|   49 | noise-2sigma   |        4 |         25 |  51.6795 |       50.2816 |  0.00188739 |
|   50 | noise-2sigma   |        4 |         50 |  49.6868 |       49.6307 |   0.0204336 |
|   51 | noise-2sigma   |        5 |        100 |  49.6541 |       49.5236 |   0.0672336 |
|   52 | noise-2sigma   |        5 |         25 |  51.4279 |       50.2023 | 0.000259294 |
|   53 | noise-2sigma   |        5 |         50 |  49.6855 |       49.5601 |         0.1 |
|   54 | noise-2sigma   |        6 |        100 |  49.6391 |       49.5414 |         0.1 |
|   55 | noise-2sigma   |        6 |         25 |  49.6603 |       49.5188 |         0.1 |
|   56 | noise-2sigma   |        6 |         50 |  49.6062 |       49.5491 |    0.030392 |
|   57 | noise-2sigma   |        7 |        100 |  49.6592 |       49.6108 |   0.0204336 |
|   58 | noise-2sigma   |        7 |         25 |  49.6097 |       49.5407 |    0.030392 |
|   59 | noise-2sigma   |        7 |         50 |  49.6163 |       49.5114 |   0.0452035 |
|   60 | noise-2sigma   |        8 |        100 |  51.3145 |       49.9485 | 7.88046e-05 |
|   61 | noise-2sigma   |        8 |         25 |  49.6595 |       49.5381 |         0.1 |
|   62 | noise-2sigma   |        8 |         50 |  49.6278 |       49.5279 |         0.1 |



### Some discussions

1. Answering (a): Yes, `ensemble_rmse` are almost always less than `avg_rmse`, indicating that ensemble method does reduce prediction error.
2. Answering (b): For `1sigma`, `n_ensemble`=100 not always better than `n_ensemble`=25, for example, comparing row index=30 versus row index=31.
3. Answering (c): (a), (b) still hold within different noise levels.
4. Notice that for a fixed noise level, n_hidden does not significantly affect the RMSE.
5. There seems no clear pattern in the `best_lr` with respect to the number of hidden neurons..

[Problem 2 train/pred code](https://github.com/weyltensor007/ncu-env-data-science/blob/main/Assignment_2/problem2_pred.py)

[Problem 2 evaluation code](https://github.com/weyltensor007/ncu-env-data-science/blob/main/Assignment_2/problem2_evaluation.py)



<div class="page"></div>



## 3. Visualize the regression results of Exercise 8.1 at least for the case with the Gaussian noise at 0.5 times the standard deviation of $y_{\text{signal}}$
