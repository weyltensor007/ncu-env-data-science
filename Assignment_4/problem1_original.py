import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

train_df = pd.read_csv("data/forest_training.csv")
test_df = pd.read_csv("data/forest_test.csv")


def miss_class_rate(n_cols):
    X_train = train_df.drop(columns=["class"]).iloc[:, :n_cols]
    y_train = train_df["class"]
    X_test = test_df.drop(columns=["class"]).iloc[:, :n_cols]
    y_test = test_df["class"]
    # normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # fit lda
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    # predict
    y_train_pred = lda.predict(X_train_scaled)
    y_test_pred = lda.predict(X_test_scaled)
    if n_cols == 2:
        np.save("preds/y_train_pred_12.1.npy", y_train_pred)
        np.save("preds/y_test_pred_12.1.npy", y_test_pred)
    misclassification_rate_train = (y_train != y_train_pred).mean()
    misclassification_rate_test = (y_test != y_test_pred).mean()
    return np.array([misclassification_rate_train, misclassification_rate_test]).round(4)

for n_cols in [2,4,6,27]:
    miss_rate_train, miss_rate_test = miss_class_rate(n_cols)
    print(f"n_cols={n_cols}: miss_rate_train={miss_rate_train}, miss_rate_test={miss_rate_test}")