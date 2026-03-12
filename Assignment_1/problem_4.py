import numpy as np
import pandas as pd

df = pd.read_csv("data/SWE_tele.csv")
df = df.loc[:, ["SWE", "Nino34", "PNA"]]
# standardized
df_standardized = (df - df.mean()) / df.std()

n_training = 45

np.random.seed(666)  # set random seed for reproducing simulation outcomes


def ridge_model(input_df, regularization_parameter):
    '''
    calculate RMSE_train, RMSE_validation, a0, a1, a2.
    note that when regularization_parameter=0, this model reduces to MLR
    '''
    df_train = input_df.iloc[:n_training, :]
    df_validation = input_df.iloc[n_training:, :]
    predictors_train = df_train.iloc[:, 1:].values
    predictors_validation = df_validation.iloc[:, 1:].values

    y_train = df_train["SWE"].values
    y_validation = df_validation["SWE"].values
    # add intercept
    X_train = np.column_stack(
        (np.ones(len(predictors_train)), predictors_train))
    X_validation = np.column_stack(
        (np.ones(len(predictors_validation)), predictors_validation))
    identity_matrix = np.identity(X_validation.shape[1])
    beta_hat = np.linalg.solve(
        (X_train.T @ X_train+regularization_parameter*identity_matrix), X_train.T @ y_train)
    beta_hat = np.round(beta_hat, 3)
    residuals_train = y_train - X_train@beta_hat
    residuals_validation = y_validation - X_validation@beta_hat
    # RMSE
    RMSE_train = np.sqrt((np.sum(residuals_train**2))/len(residuals_train))
    RMSE_validation = np.sqrt(
        (np.sum(residuals_validation**2))/len(residuals_validation))

    RMSE_train = round(RMSE_train, 3)
    RMSE_validation = round(RMSE_validation, 3)
    # return regularization_parameter, a0,a1,a2,a3,rmse_train,rmse_validation respectively
    return [regularization_parameter, *beta_hat, RMSE_train, RMSE_validation]


results_collector = []

for regularization_parameter in [0, 1e-5, 0.01]:
    for _ in range(100):
        # add small noise term to PNA, forming as the 3rd predictor
        noise = 0.001*np.random.normal(0, 1, len(df_standardized))
        df_standardized["PNA_noisy"] = df_standardized["PNA"] + noise
        result = ridge_model(df_standardized, regularization_parameter)
        results_collector.append(result)


results_df = pd.DataFrame(results_collector, columns=[
    "regularization_parameter",
    "a0_hat",
    "a1_hat",
    "a2_hat",
    "a3_hat",
    "RMSE_train",
    "RMSE_validation"
])


# calculate std of a0~a3
# calculate mean, std of RMSE_train/validation
statistics = (
    results_df.groupby("regularization_parameter")
    .agg({
        "a0_hat": "std",
        "a1_hat": "std",
        "a2_hat": "std",
        "a3_hat": "std",
        "RMSE_train": ["mean", "std"],
        "RMSE_validation": ["mean", "std"]
    })
).round(3)
statistics.columns = ['_'.join(col) for col in statistics.columns]
print(statistics.T.to_markdown())
