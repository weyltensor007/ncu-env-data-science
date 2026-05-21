# Assignment_4

**Student Name:** 郭忠侑

## 1. Complete Exercise 12.1 in Hsieh’s book. For subquestion (a), visualize the results to reproduce Figure 12.1(b). Make sure to label/mark correct and incorrect predictions.

I've used the sklean package to run the LDA model, the misclassification rates for both training and test data with different numbers of features are:

n_cols=2: miss_rate_train=0.1414, miss_rate_test=0.1969
n_cols=4: miss_rate_train=0.1414, miss_rate_test=0.2492
n_cols=6: miss_rate_train=0.0909, miss_rate_test=0.1969
n_cols=27: miss_rate_train=0.0202, miss_rate_test=0.1723

It seems that utilizing all features works best for this dataset!


Below is the plot of decision boundaries and scatter plots for the case of number of features=2, note that the points with red cross edges represent the wrongly classified points.

![Problem1](imgs/problem1_decision_boundaries_and_scatter.png)
