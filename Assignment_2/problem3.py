import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

test_df = pd.read_csv("data\\data_noise-0.5sigma_test.csv").rename(columns={'# x':"x", ' y':"y"})


min_ensemble_rmse_preds = np.load("preds\\noise-0.5sigma_6_50.npy")
y_min_rmse = np.mean(min_ensemble_rmse_preds, axis=0).reshape(-1)

max_ensemble_rmse_preds = np.load("preds\\noise-0.5sigma_4_100.npy")
y_max_rmse = np.mean(max_ensemble_rmse_preds, axis=0).reshape(-1)

x_test = test_df['x']
y_test = test_df['y']
y_true = np.sin(2*np.pi*x_test)

# plot results
plt.plot(x_test, y_true, label = "true", color = "black")
plt.scatter(x_test, y_test, label="test data", s=10)
plt.plot(x_test, y_min_rmse, label = "min-rmse", ls=":", color = 'red', lw=2)
plt.plot(x_test, y_max_rmse, label = "max-rmse", ls="-", color = "green")

plt.xlabel("x", size = 16)
plt.ylabel("y", rotation=0, size=16)
    

plt.legend()
plt.savefig("imgs/problem3-1.jpg")
plt.show()


fig, axes = plt.subplots(2,1)

# plot all members in min_ensemble_rmse_preds
ax = axes[0]

for _ in range(min_ensemble_rmse_preds.shape[0]):
    ax.plot(x_test, min_ensemble_rmse_preds[_])
ax.set_title("All min_ensemble_rmse_preds members")

# plot all members in max_ensemble_rmse_preds
ax = axes[1]

for _ in range(max_ensemble_rmse_preds.shape[0]):
    ax.plot(x_test, max_ensemble_rmse_preds[_])
    
ax.set_title("All max_ensemble_rmse_preds members")


plt.tight_layout()
plt.savefig("imgs/problem3-2.jpg")
plt.show()