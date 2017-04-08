import numpy as np
import logging
import random

from tqdm import *

def softmax(xs):
    e = np.exp(xs)
    return e / np.sum(e)

class LogisticRegressionModel(object):
    '''
        Linear model for logistic regression
        Optimize via Newton-Raphson method
    '''
    def __init__(self, classes):
        self.classes = classes
        self.w_k = {}

    def fit(self, sess, x_, t_, epochs=10, batch_size=32):
        # Initialize weight for each class
        for c in self.classes:
            self.w_k[c] = np.zeros(x_.shape[1:])
        
        # Iterative optimize
        num_sample = len(x_)
        indices = range(num_sample)
        for epoch in xrange(epochs):
            for begin in xrange(0, num_sample, batch_size):
                end = min(num_sample, begin + batch_size)
                x_train = x_[begin:end]
                t_train = t_[begin:end]
                self._optimize(sess, x_train, t_train)
                   
    def eval(self, sess, x_, t_):
        return np.sum(np.asarray(t_ - self.test(sess, x_))**2) / (2 * len(x_))
                               
    def test(self, sess, x_):
        return np.asmatrix(x_) * self.w

    def _optimize(self, sess, x_, t_):
        pass
        
 
class ProbabilisticGenerativeModel(object):
    def __init__(self, num_classes):
        self.classes = np.identity(num_classes)
        self.w_k = {}
        self.w0_k = {}
    
    def fit(self, sess, x_, t_):
        '''
            param x_: training datas
            param c: classes 
        '''
        n = len(x_)
        nc = len(self.classes)

        assert len(x_.shape) == 2
        
        d = x_.shape[1:][0]
        priors = np.asmatrix(np.zeros([nc, 1]))
        means = np.asmatrix(np.zeros([nc, d]))
        sigma = np.asmatrix(np.zeros([d, d]))
    
        for class_i, i in zip(self.classes, range(nc)):
            x_i = x_[t_[:, i] == 1]
            n_i = len(x_i)
            priors[i] = float(n_i) / n
            means[i] = np.mean(x_i, axis=0)
            sigma = (float(n_i) / n) * ((x_i - means[i]).T * (x_i - means[i])) / float(n_i)
        sigma_inv = np.linalg.inv(sigma)
        
        for i in xrange(nc):
            self.w_k[i] = sigma_inv * means[i].T
            self.w0_k[i] = (-1.0 / 2) * means[i] * sigma_inv * means[i].T + np.log(priors[i])
    
    def test(self, sess, x_):
        n = len(x_)
        m = len(self.classes)

        a_k = np.zeros([n, m])
        x_mat = np.matrix(x_)
        for i in xrange(m):
            a_k[:, i] = np.squeeze(np.asarray(x_mat * self.w_k[i] + self.w0_k[i]))
        a_k = softmax(a_k)
        y = np.asarray(self.classes)[a_k.argmax(axis=1)]
        return y

    def eval(self, sess, x_, t_):
        y = self.test(sess, x_)
        acc = float(np.equal(y.argmax(axis=1), t_.argmax(axis=1)).sum()) / len(t_) 
        return acc

if __name__ == '__main__':
    from util import one_hot
    from preprocess import Preprocessor
    import sys
 
    X = []
    Y = []
    categories = [int(c) for c in sys.argv[2].split(',')]
    num_classes = len(categories)
    for f, c in zip(sys.argv[1].split(','), categories): 
        c = int(c)
        x = np.matrix(np.load(f))
        y = np.tile(one_hot(num_classes, c), [len(x), 1])
        X.append(x)
        Y.append(y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    
    N = len(X)
    inds = range(N)
    np.random.shuffle(inds)

    n_train = int(0.8 * N)
   
    X = X[inds]
    Y = Y[inds]
 
    print ('PCA...')
    X_normal = Preprocessor().normalize(X, 255.0)
    X_phi = Preprocessor().pca(X_normal, k=2)
    print ('PCA shape = ', X_phi.shape)
    
    print ('Probablistic Generative Model...')
    
    sess = None
    model = ProbabilisticGenerativeModel(num_classes)
    model.fit(sess, X_phi[:n_train], Y[:n_train])
    model.eval(sess, X_phi[n_train:], Y[n_train:])
