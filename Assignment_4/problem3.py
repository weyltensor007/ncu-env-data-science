import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel
)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# =========================================================
# 1. Generate synthetic data
# =========================================================

np.random.seed(42)

n_samples = 40

X_train = np.linspace(0, 4*np.pi, n_samples)

# noise
noise = 0.5 * np.random.normal(0, 1, n_samples)

y_train = np.sin(X_train) + noise

# sklearn input shape = (n_samples, n_features)
X_train = X_train.reshape(-1, 1)

# prediction domain
X_test = np.linspace(0, 8*np.pi, 1000).reshape(-1, 1)

# true function
y_true = np.sin(X_test.ravel())

# =========================================================
# 2. Ridge Regression
# =========================================================

from sklearn.pipeline import Pipeline

ridge_model = Pipeline([
    ("ridge", Ridge())
])

ridge_param_grid = {
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

ridge_grid = GridSearchCV(
    ridge_model,
    ridge_param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

ridge_grid.fit(X_train, y_train)

best_ridge = ridge_grid.best_estimator_

y_ridge = best_ridge.predict(X_test)

# =========================================================
# 3. Kernel Ridge Regression
# =========================================================
#
# use RBF kernel because:
# - sine function is smooth
# - local similarity matters
# - RBF handles nonlinear patterns well
#

krr = KernelRidge(kernel="rbf")

krr_param_grid = {
    "alpha": [1e-3, 1e-2, 1e-1, 1],
    "gamma": [0.01, 0.1, 1, 10]
}

krr_grid = GridSearchCV(
    krr,
    krr_param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

krr_grid.fit(X_train, y_train)

best_krr = krr_grid.best_estimator_

y_krr = best_krr.predict(X_test)

# =========================================================
# 4. Gaussian Process Regression
# =========================================================
#
# Kernel choice:
#
# RBF + WhiteKernel
#
# RBF:
#   smooth latent function
#
# WhiteKernel:
#   models observation noise
#
# I've used only the RBF, but it fitted the noise too closely, after consulting ChatGPT,
# it suggested that I can utilize the WhiteKernel
# note that we don't need cv to tune hyperparameters in GP
# it's done within the functionality, all we need is to assign initial guess of hyperparameters

gpr_kernel = (
    RBF(length_scale=1.0)
    + WhiteKernel(noise_level=0.25)
)

gpr = GaussianProcessRegressor(
    kernel=gpr_kernel,
    n_restarts_optimizer=10,
    random_state=42
)

gpr.fit(X_train, y_train)

y_gpr, y_std = gpr.predict(X_test, return_std=True)

# =========================================================
# 5. Evaluate on training points
# =========================================================

ridge_train_pred = best_ridge.predict(X_train)
krr_train_pred = best_krr.predict(X_train)
gpr_train_pred = gpr.predict(X_train)

ridge_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
krr_rmse = np.sqrt(mean_squared_error(y_train, krr_train_pred))
gpr_rmse = np.sqrt(mean_squared_error(y_train, gpr_train_pred))

print("===== RMSE on Training Data =====")
print(f"Ridge Regression RMSE: {ridge_rmse:.4f}")
print(f"Kernel Ridge RMSE:     {krr_rmse:.4f}")
print(f"GPR RMSE:              {gpr_rmse:.4f}")

print("\n===== Best Hyperparameters =====")
print("Ridge:", ridge_grid.best_params_)
print("Kernel Ridge:", krr_grid.best_params_)
print("GPR kernel after optimization:")
print(gpr.kernel_)

# =========================================================
# 6. Visualization
# =========================================================

plt.figure(figsize=(14, 8))

# true function
plt.plot(
    X_test,
    y_true,
    linestyle="--",
    label="True sin(x)"
)

# training data
plt.scatter(
    X_train,
    y_train,
    s=50,
    label="Training Data"
)

# Ridge
plt.plot(
    X_test,
    y_ridge,
    label="Ridge Regression"
)

# Kernel Ridge
plt.plot(
    X_test,
    y_krr,
    label="Kernel Ridge Regression"
)

# GPR mean
plt.plot(
    X_test,
    y_gpr,
    label="Gaussian Process Regression"
)

# GPR uncertainty
plt.fill_between(
    X_test.ravel(),
    y_gpr - 2*y_std,
    y_gpr + 2*y_std,
    alpha=0.2,
    label="GPR ±2σ"
)

plt.xlim(0, 8*np.pi)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ridge vs Kernel Ridge vs Gaussian Process Regression")

plt.legend()
plt.grid(True)

plt.show()
