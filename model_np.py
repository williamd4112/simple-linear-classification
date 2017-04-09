import numpy as np
import logging
import random

from tqdm import *
from sklearn import linear_model

def softmax(a):
    e = np.exp(a)
    s = e.sum(axis=1)[:, np.newaxis]

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
        D = x_.shape[1:][0]
        K = self.num_classes
        self.w_k = np.asmatrix(np.zeros([K, D]))
        
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
                if self.optimizer == 'sgd':
                    self._optimize_gradient_descent(sess, x_train, t_train)
                else:
                    self._optimize(sess, x_train, t_train)
            acc = self.eval(sess, x_, t_)
            logging.info('Training accuracy = %f' % acc)
        
    def eval(self, sess, x_, t_):
        y = np.asarray(self.test(sess, x_))
        acc = float(np.equal(y.argmax(axis=1), t_.argmax(axis=1)).sum()) / len(t_) 
        return acc
                               
    def test(self, sess, x_):
        return softmax(np.asarray(np.asmatrix(x_) * np.asmatrix(self.w_k).T))

    def _optimize_gradient_descent(self, sess, x_, t_): 
        self.w_k = self.w_k - self.lr * self._gradient(sess, x_, t_)

    def _optimize(self, sess, x_, t_):
        D = x_.shape[1:][0]
        N = len(x_)
        K = self.num_classes

        X = np.asmatrix(x_)
        T = np.asmatrix(t_)
        Y = np.asmatrix(self.test(sess, x_)) 
        grad = self._gradient(sess, x_, t_).reshape([K * D, 1])
        H = self._hessian(sess, X, Y, T)
        H_inv = np.linalg.inv(H)
       
        W_old = np.asmatrix(np.reshape(self.w_k, [K * D, 1]))
        W_new = W_old - H_inv * grad
        self.w_k = W_new.reshape([K, D])
    
    def _gradient(self, sess, x_, t_):
        Y = np.asmatrix(self.test(sess, x_))
        grad = (x_.T * (Y - t_)).T
        return grad
    
    def _hessian(self, sess, x_, y_, t_):
        N = len(x_)
        D = x_.shape[1:][0]
        C = len(self.classes)
        I = np.identity(C)
        X = np.asmatrix(x_)
        y = np.asarray(y_)
        Y = np.asmatrix(y_)
        H = np.zeros([D*C, D*C])
        
        for k in xrange(C):
            for j in xrange(C):
                B = np.zeros([D, D])
                for n in xrange(N):
                    B += Y[n,k] * (I[j,k] - Y[n,j]) * (X[n].T * X[n])
                H[k*D:(k+1)*D, j*D:(j+1)*D] = B
        return H
 
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

