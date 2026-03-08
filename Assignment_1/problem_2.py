import numpy as np
import pandas as pd

df = pd.read_csv(r"data\Milwaukee_wind_direction_ozone.csv")


y = df["ozone"].values
theta = df["wind_direction"].values
cos = np.cos(2*np.pi*theta/360)
sin = np.sin(2*np.pi*theta/360)

theta_add_60 = (theta + 60) % 360

cos_add_60 = np.cos(2*np.pi*theta_add_60/360)
sin_add_60 = np.sin(2*np.pi*theta_add_60/360)

predictors = {
    "a-i": theta,
    "a-ii": np.column_stack((cos, sin)),
    "a-iii": cos,
    "a-iv": sin,
    "b-i": theta_add_60,
    "b-ii": np.column_stack((cos_add_60, sin_add_60)),
    "b-iii": cos_add_60,
    "b-iv": sin_add_60
}


def calculate_rmse(predictor, y):
    # add intercept
    X = np.column_stack((np.ones(len(predictor)), predictor))

    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)

    y_hat = X @ beta_hat

    error = y-y_hat
    rmse = np.sqrt(np.sum(error**2)/len(y))
    return round(rmse, 3)


results = {}
for id, predictor in predictors.items():
    rmse = calculate_rmse(predictor, y)
    row, col = id.split("-")
    if row not in results:
        results[row] = {}
    results[row][col] = rmse

print(pd.DataFrame(results).to_markdown())
