import numpy as np
import numpy.matlib
import csv

from tqdm import *

from numpy import vstack,array
from numpy.random import rand

class Preprocessor(object):
    def pca(self, X, k):
        '''
            Calculate covariance matrix
        '''
        cov_mat = np.cov(X.T)
        
        '''
            Calculate eigen vector
        '''
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
        
        '''
            Sort by eigen value
        '''
        eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_pairs = np.array([eig_pairs[i][1] for i in xrange(k)])
        
        return X.dot(eig_pairs.T), eig_pairs
    
    def lda(self, X, T, d):
        N, M = X.shape
        K = T.shape[1]
        
        means = np.array([np.mean(X[T.argmax(axis=1) == k], axis=0) for k in xrange(K)])
        S_W = np.zeros([M, M])

        for k, mean in zip(range(K), means):
            class_sc_mat = np.zeros([M, M])   
            x_k = X[T.argmax(axis=1) == k]
            S_W += ((x_k - mean).T.dot(x_k - mean))
        
        mean_all = np.mean(X, axis=0)
        S_B = np.zeros([M, M])
        for k, mean in zip(range(K), means):
            n = len(X[T.argmax(axis=1) == k,:])
            S_B += n * (mean - mean_all).dot((mean - mean_all))

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        for i in range(len(eig_vals)):
            eigvec_sc = eig_vecs[:,i].reshape(M, 1)   
        
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        eig_pairs = np.hstack(([eig_pairs[i][1].reshape(M,1) for i in xrange(d)])).T.astype(np.float32)
        
        return X.dot(eig_pairs.T), eig_pairs        

    def normalize(self, X):
        obs = X
        std_dev = np.std(obs, axis=0)
        return obs / std_dev, std_dev
