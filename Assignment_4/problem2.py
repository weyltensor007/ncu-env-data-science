import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import combinations
from collections import defaultdict

# =========================================================
# 1. Load data
# =========================================================
train_df = pd.read_csv("data/forest_training.csv")
test_df = pd.read_csv("data/forest_test.csv")

n_cols = 2
X_train = train_df.drop(columns=["class"]).iloc[:, :n_cols]
y_train = train_df["class"]

X_test = test_df.drop(columns=["class"]).iloc[:, :n_cols]
y_test = test_df["class"]

# =========================================================
# 2. Normalize
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 3. Train One-vs-One SVMs
# =========================================================
classes = np.unique(y_train)
svm_models = {}

for c1, c2 in combinations(classes, 2):

    idx = (y_train == c1) | (y_train == c2)
    X_pair = X_train_scaled[idx]
    y_pair = y_train[idx]

    # binary labels
    y_pair_binary = np.where(y_pair == c1, 0, 1)

    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_pair, y_pair_binary)

    svm_models[(c1, c2)] = clf


# =========================================================
# 4. Prediction (majority voting)
# =========================================================
def predict_ovo(X):
    results = []

    for x in X:
        vote_count = defaultdict(int)

        for (c1, c2), clf in svm_models.items():
            pred = clf.predict([x])[0]

            if pred == 0:
                vote_count[c1] += 1
            else:
                vote_count[c2] += 1

        results.append(max(vote_count, key=vote_count.get))

    return np.array(results)




# =========================================================
# 5. Decision boundary plotting
# =========================================================
def plot_decision_boundary(X, y, predict_func, title):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

    # correct / incorrect
    pred = predict_func(X)
    correct = (pred == y)

    # correct points
    plt.scatter(
        X[correct, 0], X[correct, 1],
        c=y[correct],
        cmap="viridis",
        edgecolor="k",
        label="Correct"
    )
    # misclassified
    print(len(        X_train_scaled[~correct, 0]))
    plt.scatter(
        X_train_scaled[~correct, 0],
        X_train_scaled[~correct, 1],
        c=y_train.values[~correct],   # preserve class
        edgecolor="red",              # highlight using edgecolor
        linewidth=2,
        marker="X",
        s=120,
    )

    plt.title(title)
    plt.show()


# wrapper
def predict_wrapper(X):
    return predict_ovo(X)


# =========================================================
# 6. Plots
# =========================================================
plot_decision_boundary(
    X_train_scaled,
    y_train,
    predict_wrapper,
    "1 vs. 1 SVM Decision Boundary (Train)"
)