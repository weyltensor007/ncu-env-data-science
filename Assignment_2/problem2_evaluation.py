import numpy as np
import pandas as pd
import os

npys = [_.path for _ in os.scandir("preds") if _.path.endswith(".npy")]
test_csvs = [_.path for _ in os.scandir("data") if "test" in _.path]
test_dfs = [pd.read_csv(_).rename(columns={'# x':"x", ' y':"y"}) for _ in test_csvs]


def evaluate(y_preds, y_test):
    # individual RMSE
    rmses = [
        np.sqrt(np.sum((y_test - y_preds[i].reshape(-1))**2))
        for i in range(y_preds.shape[0])
    ]
    avg_rmse = np.mean(rmses)
    
    # ensemble RMSE
    ensemble_pred = np.mean(y_preds, axis=0).reshape(-1)
    
    ensemble_rmse = np.sqrt(np.sum((y_test-ensemble_pred)**2))
    
    return avg_rmse, ensemble_rmse


evaluation_results = []

for npy in npys:
    sigma, n_hidden, n_ensemble = os.path.splitext(os.path.basename(npy))[0].split("_")
    corresponding_csv_index = test_csvs.index(f"data\\data_{sigma}_test.csv")
    corresponding_test_df = test_dfs[corresponding_csv_index]
    y_test = corresponding_test_df['y']
    y_preds = np.load(npy)
    avg_rmse, ensemble_rmse = evaluate(y_preds, y_test)

    evaluation_results.append([sigma, n_hidden, n_ensemble, avg_rmse, ensemble_rmse])

# merge results
evaluation_results_df = pd.DataFrame(evaluation_results, columns=["sigma", "n_hidden", "n_ensemble", "avg_rmse", "ensemble_rmse"])
best_lr_results_df = pd.read_csv("problem2_best_lr_results.csv")
# make sure matching data types 
evaluation_results_df['n_hidden'] = evaluation_results_df['n_hidden'].astype(int)
evaluation_results_df['n_ensemble'] = evaluation_results_df['n_ensemble'].astype(int)

full_info_df = pd.merge(evaluation_results_df,best_lr_results_df,on=["sigma","n_hidden","n_ensemble"])
full_info_df = full_info_df.sort_values(["sigma", "n_hidden", "n_ensemble"], ignore_index=True)
full_info_df.to_csv("problem2_full_info.csv", index=False)
print(full_info_df.to_markdown())