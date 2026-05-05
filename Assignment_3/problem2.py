import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# read in and preprocess data

df = pd.read_csv(r"data\San_Francisco_Bay.csv")  # chl, dox, spm, sal, temp
X = df.values
n = X.shape[0] # number of data
X = X- X.sum(axis=0)/n # center the data

# do PCA, getting the A, E matrices and explained variances
pca = PCA()
pca.fit(X)
E = pca.components_.T # each row in pca.components_ represent each eigenvector^T, so we need to do transpose to get E
A = X.dot(E) # equation (9.66) in the textbook
'''
total_variance can be evaluated from eigenvalues or use attribute of pca directly
compute it from eigenvalues is helpful when calculating variance in RPCA
also, it should be noted that total_variance=(sum of eigen values)/(n-1) for sklearn, we follow that convention
'''
total_variance_by_eigenvalues = (pca.singular_values_**2).sum()/(n-1)
assert total_variance_by_eigenvalues == np.sum(pca.explained_variance_)

# calculate the percentage of variance for k=2,3,4,5 (k=truncated number of dimensions)
pca_explained_ratio = pca.explained_variance_ratio_
pov_pca = pca_explained_ratio.cumsum()[1:] # pov=percentage of variance(for fixed k=2~5)
