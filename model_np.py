import numpy as np
import logging
import random

from tqdm import *
from sklearn import linear_model

def softmax(a):
    e = np.exp(a)
    s = e.sum(axis=1)
    return e / e.sum(axis=1)[:, np.newaxis]

class LogisticRegressionModel(object):
    '''
        Linear model for logistic regression
        Optimize via Newton-Raphson method
    '''
    def __init__(self, num_classes, optimizer='netwon', lr=0.01):
        self.classes = np.identity(num_classes)
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr = lr
        self.w_k = None

    def fit(self, sess, x_, t_, epochs=1, batch_size=2400):
        assert len(x_.shape) == 2
        
        # Initialize weight for each class
        M = x_.shape[1:][0]
        K = self.num_classes
        self.w_k = np.zeros([K, M])
        
        # Iterative optimize
        num_sample = len(x_)
        indices = range(num_sample)
        sess = None
        
        for epoch in xrange(epochs):
            np.random.shuffle(indices)
            for begin in xrange(0, num_sample, batch_size):
                end = min(num_sample, begin + batch_size)
                x_train = x_[begin:end]
                t_train = t_[begin:end]
                self._optimize(sess, x_train, t_train)
            acc = self.eval(sess, x_, t_)
            logging.info('Training accuracy = %f' % acc)
        
    def eval(self, sess, x_, t_):
        y = np.asarray(self.test(sess, x_))
        t = np.asarray(t_)
        acc = float(np.equal(y.argmax(axis=1), t.argmax(axis=1)).sum()) / len(t_) 
        return acc
                               
    def test(self, sess, x_):
        x = np.asarray(x_)
        w = np.asarray(self.w_k)
        y = softmax(x.dot(w.T))
        return y

    def _optimize(self, sess, x_, t_):
        N, M = x_.shape
        K = self.num_classes
        
        x = np.asarray(x_)
        y = np.asarray(self.test(sess, x_))
        t = np.asarray(t_)

        # Calculate gradient
        grad = np.zeros([K, M])
        for j in xrange(K):
            for n in xrange(N):
                grad[j, :] += (y[n,j] - t[n,j]) * x[n]
        grad = grad.flatten()
 
        # Calculate Hessian matrix
        I = np.identity(K)
        H = np.zeros([K*M, K*M])
        for j in xrange(K):
            for k in xrange(K):
                D_wjk = np.zeros([M, M])
                for n in xrange(N):
                    D_wjk += y[n,k] * (I[k,j] - y[n,j]) * x.T.dot(x)
                H[j*(M):(j+1)*M, (k)*M:(k+1)*M] = D_wjk
        try:
            H_inv = np.linalg.pinv(H)
        except np.linalg.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        # Update weight
        w_old = self.w_k.flatten()
        w_new = w_old - self.lr * H_inv.dot(grad)
        w_new = w_new.reshape([K, M])
        self.w_k = w_new

class ProbabilisticGenerativeModel(object):
    def __init__(self, num_classes):
        self.classes = np.identity(num_classes)
        self.w_k = {}
        self.w0_k = {}
    
    def fit(self, sess, x_, t_, epoch=None, batch_size=None):
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
        sigma_inv = np.linalg.pinv(sigma)
        
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
        t = np.asarray(t_)
        acc = float(np.equal(np.argmax(y, axis=1), np.argmax(t, axis=1)).sum()) / len(t) 
        return acc

