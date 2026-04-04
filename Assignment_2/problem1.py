import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# load data
df = pd.read_csv("data/SWE_tele.csv")


# specify X, Y
X = df.loc[:, ["Nino34", "PNA"]]
Y = df.loc[:, ["SWE"]]

# split into train/test
n_train = 40
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# set random seed for reproducing simulation outcomes
np.random.seed(666)  

'''
build an ELM(extreme learning machine) model with ensembles,
and treat n_hidden_neurons as a hyperparameter to be tuned.
'''

'''
# ELM (single model)
'''
def ELM_single(X_train, Y_train, X_val, n_hidden):
    X_train = X_train.values
    Y_train = Y_train.values
    X_val = X_val.values

    n_features = X_train.shape[1]

    # random weights
    W = np.random.randn(n_features, n_hidden)
    b = np.random.randn(1, n_hidden)

    # activation
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    H_train = sigmoid(X_train @ W + b)
    H_val   = sigmoid(X_val @ W + b)

    # output weights
    beta = np.linalg.pinv(H_train) @ Y_train

    # prediction
    Y_pred = H_val @ beta

    return Y_pred

'''
n-fold CV tuning
'''
def tune_n_hidden_cv(X, Y, hidden_list, n_folds=5, n_repeats=5):
    X = X.reset_index(drop=True) # reset index for indexing
    Y = Y.reset_index(drop=True)

    n = len(X)
    fold_size = n // n_folds

    results = {}

    for n_hidden in hidden_list:
        rmses_all = []

        for repeat in range(n_repeats): # repeat to lower randomness
            rmses = []

            for k in range(n_folds):
                start = k * fold_size
                end = (k + 1) * fold_size if k < n_folds - 1 else n

                X_val = X.iloc[start:end]
                Y_val = Y.iloc[start:end]

                X_tr = pd.concat([X.iloc[:start], X.iloc[end:]])
                Y_tr = pd.concat([Y.iloc[:start], Y.iloc[end:]])

                Y_pred = ELM_single(X_tr, Y_tr, X_val, n_hidden)

                rmse = np.sqrt(np.mean((Y_val.values - Y_pred) ** 2))
                rmses.append(rmse)

            rmses_all.append(np.mean(rmses))

        results[n_hidden] = round(np.mean(rmses_all),3)

    best_n_hidden = min(results, key=results.get)

    return best_n_hidden, results

'''
# ELM Ensembles
'''
def ELM_ensembles(X_train, Y_train, X_test, Y_test, n_hidden_neurons, n_ensembles):
    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values

    n_samples, n_features = X_train.shape

    preds = []

    for i in range(n_ensembles):
        # random weights
        W = np.random.randn(n_features, n_hidden_neurons)
        b = np.random.randn(1, n_hidden_neurons)

        # activation
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        H_train = sigmoid(X_train @ W + b)
        H_test  = sigmoid(X_test @ W + b)

        # output weights
        beta = np.linalg.pinv(H_train) @ Y_train

        # prediction
        Y_pred = H_test @ beta
        preds.append(Y_pred)

    preds = np.array(preds)
    Y_pred_mean = np.mean(preds, axis=0)

    # RMSE
    rmse = np.sqrt(np.mean((Y_test - Y_pred_mean) ** 2))

    # Pearson correlation
    corr = np.corrcoef(Y_test.flatten(), Y_pred_mean.flatten())[0, 1]

    return round(rmse,3), round(corr,3)

'''
MLR model(outputting rmse, corr)
'''
def MLR_model(X_train, Y_train, X_test, Y_test):
    # add intercept
    X_train = X_train.values
    Y_train = Y_train.values.ravel()
    X_test = X_test.values
    Y_test=Y_test.values.ravel()
    # fit beta_hat
    X = np.column_stack((np.ones(len(X_train)), X_train))
    beta_hat = np.linalg.solve(X.T @ X, X.T @ Y_train)
    # predict and calculate rmse & corr
    Y_pred = np.column_stack((np.ones(len(X_test)), X_test)) @ beta_hat
    error = Y_test - Y_pred
    rmse = np.sqrt(np.sum(error**2)/len(error))
    corr = np.corrcoef(Y_test, Y_pred)[0,1]
    return round(rmse,3), round(corr,3)


# MAIN
# 1. tuning
hidden_list = [5, 10, 20, 50, 100]
best_h, cv_results = tune_n_hidden_cv(X_train, Y_train, hidden_list)
print("CV rmse results:", cv_results)
print("Best n_hidden:", best_h)

# 2. final model (ensemble vs. MLR)
rmse_ELM_ensembles, corr_ELM_ensembles = ELM_ensembles(
    X_train, Y_train,
    X_test, Y_test,
    n_hidden_neurons=best_h,
    n_ensembles=100 # set a fixed ensemble numbers
)
rmse_MLR, corr_MLR = MLR_model(X_train, Y_train, X_test, Y_test)
print(f"rmse_ELM_ensembles={rmse_ELM_ensembles}, rmse_MLR={rmse_MLR}")
print(f"corr_ELM_ensembles={corr_ELM_ensembles}, corr_MLR={corr_MLR}")