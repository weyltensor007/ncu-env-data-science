import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# read in and preprocess data

df = pd.read_csv(r"data\San_Francisco_Bay.csv")  # chl, dox, spm, sal, temp
X = df.values
n = X.shape[0] # number of data
X = (X- X.mean(axis=0))/X.std(axis=0) # center and standardize the data

# do full components PCA in order to get total variance of data
pca = PCA()
pca.fit(X)
total_variance = pca.explained_variance_.sum()
print("sklearn:", pca.explained_variance_ratio_)
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


# start doing PCA/RPCA for different numbers of components k
results = []
for k in range(1,X.shape[1]+1):
    # do PCA, getting the A, E matrices and explained variances
    pca = PCA(n_components=k)
    pca.fit(X)
    E = pca.components_.T # each row in pca.components_ represent each eigenvector^T, so we need to do transpose to get E
    A = X.dot(E) # equation (9.66) in the textbook
    explained_variance_PCA = calculate_explained_variance_ratio(A)
    
    # A-frame rotation
    rotation_matrix_A_frame = get_rotation_matrix_of_RPCA(target_matrix=E)
    A_A_frame = A.dot(rotation_matrix_A_frame)
    explained_variance_A_frame = calculate_explained_variance_ratio(A_A_frame)
    # E-frame rotation
    rotation_matrix_E_frame = get_rotation_matrix_of_RPCA(target_matrix=A)
    A_E_frame = A.dot(rotation_matrix_E_frame)
    explained_variance_E_frame = calculate_explained_variance_ratio(A_E_frame)

    # results.append([explained_variance_PCA, explained_variance_E_frame, explained_variance_A_frame])
    print(f"k={k}\n PCA={explained_variance_PCA}\n E={explained_variance_E_frame}\n A={explained_variance_A_frame}")



