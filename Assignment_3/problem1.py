import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# 1. loading data
# =========================
df = pd.read_csv(r"data\Exer_9.4.csv")

# =========================
# 2. standardize
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================
# 3. PCA
# =========================
pca = PCA()
Z = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_

print("\n=== Explained Variance Ratio ===")
print(explained_var) # it seems that first two components are enough to explain almost all variances

# =========================
# 4. Loadings(eigen-vectors)
# =========================
components = pca.components_

loadings = pd.DataFrame(
    components.T,
    columns=[f"e_{i+1}" for i in range(components.shape[0])],
    index=df.columns
)

print("\n=== Loadings ===")
print(loadings)

# =========================
# 5. reducing the dimension to 2 (as indicated in the explained_var)
# =========================
Z_2D = Z[:, :2]

Z_df = pd.DataFrame(Z_2D, columns=["PC1", "PC2"])

# =========================
# 6. visualize transformed coordinates after PCA_2D
# =========================
plt.figure()

plt.scatter(Z_2D[:, 0], Z_2D[:, 1])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection (2D)")

plt.axhline(0)
plt.axvline(0)

plt.grid(True)

plt.show()