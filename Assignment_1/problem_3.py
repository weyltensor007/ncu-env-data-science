import numpy as np
import pandas as pd
from scipy.stats import t

df = pd.read_csv("data/SWE_tele.csv")

n_training = 45
y_col = "SWE"

# split df into training and validation
df_training = df[:n_training]
df_validation = df[n_training:]


def calculate_p_values_and_RMSE(predictors_cols):
    predictors = df_training.loc[:, predictors_cols]
    y_train = df_training.loc[:, y_col]
    y_validation = df_validation.loc[:, y_col]
    # add intercept
    X_train = np.column_stack((np.ones(len(predictors)), predictors))
    X_validation = np.column_stack(
        (np.ones(len(df_validation)), df_validation.loc[:, predictors_cols]))
    # analytic form of MLR
    beta_hat = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
    residuals_train = y_train - X_train @ beta_hat
    residuals_validation = y_validation - X_validation@beta_hat
    # RMSE
    RMSE_train = np.sqrt((np.sum(residuals_train**2))/len(residuals_train))
    RMSE_validation = np.sqrt(
        (np.sum(residuals_validation**2))/len(residuals_validation))
    # variance
    n, p = X_train.shape
    sigma2_hat = (residuals_train.T @ residuals_train) / (n - p)

    # covariance
    cov_beta = sigma2_hat * np.linalg.inv(X_train.T @ X_train)

    # standard errors
    se_beta = np.sqrt(np.diag(cov_beta))

    # t statistics
    t_stats = beta_hat.flatten() / se_beta

    # p value
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=n-p))

    # round to 2 digits
    p_values = np.round(p_values[1:], 2)  # neglect the intercept term
    RMSE_train = round(RMSE_train, 2)
    RMSE_validation = round(RMSE_validation, 2)
    return p_values, RMSE_train, RMSE_validation


predictors_cols = ["Nino34", "PNA", "NAO", "AO"]
iteration = 1
while len(predictors_cols) >= 1:
    p_values, RMSE_train, RMSE_validation = calculate_p_values_and_RMSE(
        predictors_cols)
    print("-"*50+f"round:{iteration}"+"-"*50)
    for p_value, predictor in zip(p_values, predictors_cols):
        print(f"{predictor}, p-value={p_value}")
    id_of_max_p_value = np.argmax(p_values)
    predictor_to_drop = predictors_cols[id_of_max_p_value]
    print(
        f"drop:{predictor_to_drop}, RMSE_train={RMSE_train}, RMSE_validation={RMSE_validation}")
    predictors_cols.pop(id_of_max_p_value)
    iteration += 1
