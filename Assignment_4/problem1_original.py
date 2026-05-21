import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    if n_cols == 2: # plot decision boundaries
        x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
        y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        
        Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)

        # -------------------------
        # correct / wrong mask
        # -------------------------
        correct = (y_train.values == y_train_pred)

        # -------------------------
        # 1. correct points
        # -------------------------
        plt.scatter(
            X_train_scaled[correct, 0],
            X_train_scaled[correct, 1],
            c=y_train.values[correct],
            edgecolor="k",
        )

        # -------------------------
        # 2. misclassified points（preserve class color, highlight wrong class）
        # -------------------------
        plt.scatter(
            X_train_scaled[~correct, 0],
            X_train_scaled[~correct, 1],
            c=y_train.values[~correct],   # preserve class
            edgecolor="red",              # highlight using edgecolor
            linewidth=2,
            marker="X",
            s=120,
        )

        plt.xlabel("Feature 1 (scaled)", size =14)
        plt.ylabel("Feature 2 (scaled)", size=14)
        plt.title("LDA Decision Boundary + Misclassification",size=18)
        plt.show()
    misclassification_rate_train = (y_train != y_train_pred).mean()
    misclassification_rate_test = (y_test != y_test_pred).mean()
    return np.array([misclassification_rate_train, misclassification_rate_test]).round(4)

for n_cols in [2,4,6,27]:
    miss_rate_train, miss_rate_test = miss_class_rate(n_cols)
    print(f"n_cols={n_cols}: miss_rate_train={miss_rate_train}, miss_rate_test={miss_rate_test}")