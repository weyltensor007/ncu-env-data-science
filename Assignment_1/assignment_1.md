# Assignment_1

**Student Name:** 郭忠侑

## 1. Reproduce the figure for Q3 in the Pre-course Quiz.

This is my resulting image:

![Problem1](imgs/Assignment_1_problem1.png)

[Problem 1 code](https://github.com/weyltensor007/ncu-env-data-science/blob/main/Assignment_1/problem_1.py)

<div class="page"></div>

## 2. Exercise 5.7 in Hsieh’s book

![Problem2](imgs/5.7.png)

Table of RMSE for different cases:

|      |      a |      b |
| :--- | -----: | -----: |
| i    | 20.396 | 28.220 |
| ii   | 19.765 | 19.765 |
| iii  | 28.194 | 23.636 |
| iv   | 20.587 | 23.458 |

### Caveat

When dealing with the case of $\theta'=\theta+60$, we need to do one more operation, that is the modular operation which makes the transformed angles lie in the correct range: $[0,360)$. More specifically, we let $\theta'=(\theta+60) \mod 360$.

### Discussion of the results

- although case a-i is not bad within case a, but using the parameter $\theta$ directly is unstable when adding a constant to it, we can see this effect in case b-i, which is the worst within case b.
- the RMSE values of case a-ii and b-ii are equivalent, this is not a coincidence, in fact, it is related to the fact that when one perform an invertible linear transformation on the predictor variables, the RMSE is invariant under such transformation. we can prove this fact in the following section.

### RMSE is invariant under an invertible linear transformation on the predictor variables

#### Why do we care about the linear transformation on the predictor variables?

Consider the predictor variables in case ii, denote $k=60^{\circ}$ and let $\theta$ being radians of the raw angular data, $\theta'=\theta+k$ being the shifted angle, then we have four predictors:

- Predictor 1 of case a:  $X^{a}_1=\cos(\theta)$
- Predictor 2 of case a:  $X^{a}_2=\sin(\theta)$
- Predictor 1 of case b:  $X^{b}_1=\cos(\theta+k)$
- Predictor 2 of case b:  $X^{b}_2=\sin(\theta+k)$

By trigonometric identities:

$$
\begin{cases}
X^{b}_1=\cos(k)\cdot X^{a}_1-\sin(k)\cdot X^{a}_2\\\\
X^{b}_2=\sin(k)\cdot X^{a}_1+\cos(k)\cdot X^{a}_2
\end{cases}
$$

which means we are essentially dealing with **linear transformation of the predictor variables**!

#### Transformation of the estimated parameters

Recall that for a model $y=X\beta$, we have:

- the estimated parameter $\hat{\beta}=(X^TX)^{-1}X^T y$

- the predicted response $\hat{y}=X\hat{\beta}$

Denote the transformed predictors $X'=TX$, in which the matrix $T$ encapsulates the transformation rule, the model now becomes $y=X'\beta'$, and similarly:

- the estimated parater $\hat{\beta'}=(X'^TX')^{-1}X'^T y$
- the predicted response $\hat{y'}=X'\hat{\beta'}$

By matrix algebra we can establish the relation between $\hat{\beta}$ and $\hat{\beta'}$

$$
\begin{align*}
    \hat{\beta'}&=(X'^TX')^{-1}X'^T y\\\\
    &=[(T^TX^TXT)^{-1}]T^TX^T y\\\\
    &=[T^{-1}(X^TX)^{-1}(T^T)^{-1}]T^TX^Ty\\\\
    &=T^{-1}(X^TX)^{-1}X^Ty\\\\
    &=T^{-1}\hat{\beta}
\end{align*}
$$

then we can show that $\hat{y'}=\hat{y}$

$$
\hat{y'}=X'\hat{\beta'}=XT(T^{-1}\hat{\beta})=X\hat{\beta}=\hat{y}
$$

since $\hat{y'}=\hat{y}$, the RMSE stays the same under transformation $T$.

Note that the usually used standardization of predictors, say $z=(x-\bar{x})/s$ is also a linear transformation and thus it won't affect the RMSE either.

By the way, when use a single component(say $\sin$ or $\cos$) will not guarantee that RMSE being invariant under shifting of angles, since it may introduce another component, roughly speaking:

$$
\sin(\theta+k)=\sin(\theta)\cos(k)+\sin(k)\cos(\theta)
$$

another independent variable $\cos(\theta)$ appears, this explains the fact that in case iii and iv the RMSE is not invariant.

[Problem 2 code](https://github.com/weyltensor007/ncu-env-data-science/blob/main/Assignment_1/problem_2.py)

<div class="page"></div>

## 3. Exercise 5.8 in Hsieh’s book

![Problem3](imgs/5.8.png)


This exercise is related to the so called "Backward Elimination", and result is summarized by the table below:

| round | Nino34 | PNA  | NAO  | AO   | RMSE_train | RMSE_validation |
| ----- | ------ | ---- | ---- | ---- | ---------- | --------------- |
| 1     | 0.25   | 0.02 | 0.24 | 0.29 | 424.74     | 544.18          |
| 2     | 0.25   | 0.01 | 0.57 |      | 430.9      | 552.96          |
| 3     | 0.26   | 0.01 |      |      | 432.62     | 541.9           |
| 4     |        | 0    |      |      | 439.22     | 578.3           |

### Discussions

1. By assessing the model with lowest RMSE_validation, the best model would be round 3 which contains Nino34, PNA as predictors.
2. No, if I was asked to choose two predictors based on the p-values on round 1 model, I may choose PNA and NAO with lower p-values as predictors, which is not necessary the best one according to the criteria above.

[Problem 3 code](https://github.com/weyltensor007/ncu-env-data-science/blob/main/Assignment_1/problem_3.py)


<div class="page"></div>

## 4. Exercise 5.9 in Hsieh’s book

![Problem4](imgs/5.9.png)

The result of my simulation is summarized by the table below, note that $\lambda=0$ corresponds to MLR.
Also, I have fixed the random seed in my code as `np.random.seed(666)` in order to keep the reproducibility.

| statistics \\ $\lambda$ |     0.0 |  1e-05 |  0.01 |
| :---------------------- | ------: | -----: | ----: |
| a0_hat_std              |   0.022 |  0.015 |     0 |
| a1_hat_std              |   0.026 |  0.016 |     0 |
| a2_hat_std              | 138.125 | 90.109 | 0.265 |
| a3_hat_std              | 138.126 | 90.111 | 0.265 |
| RMSE_train_mean         |   0.812 |  0.813 | 0.822 |
| RMSE_train_std          |   0.013 |   0.01 |     0 |
| RMSE_validation_mean    |   1.035 |  1.037 |  1.03 |
| RMSE_validation_std     |   0.028 |  0.025 |     0 |

### Discussions

1. It's easy to see that for highly colinear pairs($\hat{a_2}$, $\hat{a_3}$), the variances of these estimators are pretty high and nearly equal. This phenomenon can be explain by eigen-decomposition of a symmetric matrix($X^TX$), since $x_2$, $x_3$ are nearly parallel, there exist an nearly zero eigenvalue and the corresponding eigenvector $[0,0,1,-1]^T$ such that $\text{Var}\hat{\beta}\sim(X^TX)^{-1}=\sum_{i} (1/\lambda_i)v_i v_i^T$ would produces large value at $\beta_2$ and $\beta_3$
2. Once we've raised the regularization parameter, the variance of those colinear pairs are largely suppressed, showing the effect of regularization on the problem of collinearity.
3. The effect of $\lambda=10^{-5}$ is clearly not enough, since almost all statistics are nearly equal, so it's reasonable to use higher $\lambda$ for highly colinear predictors. Regularization suppresses variance because it inflates the small eigenvalues of $X^{T}X$, preventing the inverse matrix from exploding in directions caused by collinearity.
4. Although regularization reduced the variance of the coefficient estimates, the mean RMSE for both the training and validation sets remained largely unchanged.

[Problem 4 code](https://github.com/weyltensor007/ncu-env-data-science/blob/main/Assignment_1/problem_4.py)