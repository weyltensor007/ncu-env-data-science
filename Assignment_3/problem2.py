import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

cmap = plt.get_cmap('tab10')

# read in and preprocess data
df = pd.read_csv(r"data\San_Francisco_Bay.csv")  # chl, dox, spm, sal, temp
X = df.values
n = X.shape[0] # number of data
X = (X- X.mean(axis=0))/X.std(axis=0) # center and standardize the data

# do full components PCA in order to get total variance of data
pca = PCA()
pca.fit(X)
total_variance = pca.explained_variance_.sum()

def calculate_explained_variance_ratio(A):
    '''
    A means the projected component in some basis,
    A <-> E
    A_tilde=AR <-> E_tilde=ER
    '''
    explained_variance = np.diag((A.T).dot(A))/(A.shape[0]-1)
    return np.round(explained_variance/total_variance,3)


def get_dL_dR(rotation_matrix, target_matrix):
    '''
    dL_dR(rotation_matrix, target_matrix)
    for E-frame rotation: target_matrix = A
    for A-frame rotation: target_matrix = E
    '''
    # define some basic terms
    M = target_matrix
    R = rotation_matrix
    M_tilde = M.dot(R)
    diagonal_term = np.diag(np.diag((M_tilde.T).dot(M_tilde)))
    
    first_term = M_tilde**3 # element wise cubed
    second_term = M_tilde.dot(diagonal_term)/M_tilde.shape[0]
    return (M.T).dot(first_term-second_term)


# RPCA algorithm
def get_rotation_matrix_of_RPCA(target_matrix, tol=1e-6, max_iter=100):
    # init rotation matrix as identity matrix
    rotation_matrix = np.eye(target_matrix.shape[1])
    s = 0 # sum of singular values of dL_dR
    
    # start looping
    for _ in range(max_iter):
        # calculate dL_dR matrix
        dL_dR = get_dL_dR(rotation_matrix, target_matrix)
        # get svd of dL_dR in order to update rotation matrix
        U, sigma, VT = np.linalg.svd(dL_dR)
        # update rotation matrix
        rotation_matrix = U.dot(VT)
        # calculate sum(singular values) and check for break
        s_new = sigma.sum()
        if s!=0 and s_new<s*(1+tol): # if no improvement then break
            break
        s = s_new
    return rotation_matrix

fig1, axes1 = plt.subplots(3, 4, figsize=(10, 10), constrained_layout=True) # for plotting eigenvector components
fig2, axes2 = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True) # for plotting eigenvector components

x = np.arange(1,6)
for ax1, ax2 in zip(axes1.flat,axes2.flat):
    ax1.set_xticks(x)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(-1,1)

    ax2.set_xticks(x)
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(0, total_variance)
# start doing PCA/RPCA for different numbers of components k
# in the mean while, plot components of eigenvectors and explained variance ratio

for k in range(2,X.shape[1]+1):
    # ordinary PCA, getting the A, E matrices and explained variances
    pca = PCA(n_components=k)
    pca.fit(X)
    E = pca.components_.T # each row in pca.components_ represent each eigenvector^T, so we need to do transpose to get E
    A = X.dot(E) # equation (9.66) in the textbook
    explained_variance_PCA = calculate_explained_variance_ratio(A)
    ## plot components of each column vector in E
    for i in range(E.shape[1]):
        if k != X.shape[1]:
            axes1[0, k-2].plot(x, E[:,i], color=cmap(i))
        else:
            axes1[0, k-2].plot(x, E[:,i], color=cmap(i),label=f"e_{i+1}")
        axes1[0,k-2].set_title(f"k={k},PCA")
    # A-frame rotation
    rotation_matrix_A_frame = get_rotation_matrix_of_RPCA(target_matrix=E)
    A_A_frame = A.dot(rotation_matrix_A_frame)
    E_A_frame = E.dot(rotation_matrix_A_frame)
    explained_variance_A_frame = calculate_explained_variance_ratio(A_A_frame)
    ## plot components of each column vector in E_A_frame
    for i in range(E_A_frame.shape[1]):
        if k != X.shape[1]:
            axes1[1, k-2].plot(x, E_A_frame[:,i], color=cmap(i))
        else:
            axes1[1, k-2].plot(x, E_A_frame[:,i], color=cmap(i),label=f"e_{i+1}")
        axes1[1,k-2].set_title(f"k={k},A-frame")
    # E-frame rotation
    rotation_matrix_E_frame = get_rotation_matrix_of_RPCA(target_matrix=A)
    A_E_frame = A.dot(rotation_matrix_E_frame)
    E_E_frame = E.dot(rotation_matrix_E_frame)
    explained_variance_E_frame = calculate_explained_variance_ratio(A_E_frame)
    ## plot components of each column vector in E_E_frame
    for i in range(E_E_frame.shape[1]):
        if k != X.shape[1]:
            axes1[2, k-2].plot(x, E_E_frame[:,i], color=cmap(i))
        else:
            axes1[2, k-2].plot(x, E_E_frame[:,i], color=cmap(i),label=f"e_{i+1}")
        axes1[2,k-2].set_title(f"k={k},E-frame")

    print(f"k={k}")
    print(f"PCA_variance={explained_variance_PCA}")
    print(f"E-frame_variance={explained_variance_E_frame}")
    print(f"A-frame_variance={explained_variance_A_frame}")


handles, labels = axes1[0,3].get_legend_handles_labels()

# single shared legend
fig1.legend(handles, labels,
            loc='upper right')

plt.show()