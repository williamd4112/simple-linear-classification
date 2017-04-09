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
    def __init__(self, n_classes=1, lr=1.0, tolerance=0.1):
        self.classes = np.identity(n_classes)
        self.n_classes = n_classes
        self.lr = lr
        self.w_k = None
        self.tolerance = tolerance
   
    def save(self, path):
        np.save(path, self.w_k)
        logging.info('Saving model to %s success [K = %d, M = %d]' % (path, self.w_k.shape[0], self.w_k.shape[1]))

    def load(self, path):
        self.w_k = np.load(path)
        self.n_classes = self.w_k.shape[0]
        self.n_features = self.w_k.shape[1]
        logging.info('Loading model from %s success [K = %d, M = %d]' % (path, self.w_k.shape[0], self.w_k.shape[1]))

    def fit(self, sess, x_, t_, epochs=1, batch_size=2400):
        assert len(x_.shape) == 2
        
        # Initialize weight for each class
        M = x_.shape[1:][0]
        K = self.n_classes
        self.w_k = np.zeros([K, M])
        self.n_classes = self.w_k.shape[0]
        self.n_features = self.w_k.shape[1]

        # Iterative optimize
        num_sample = len(x_)
        indices = range(num_sample)
        sess = None
        
        for epoch in xrange(epochs):
            np.random.shuffle(indices)
            for begin in xrange(0, num_sample, batch_size):
                end = min(num_sample, begin + batch_size)
                x_train = x_[indices[begin:end]]
                t_train = t_[indices[begin:end]]
                self._optimize(sess, x_train, t_train)
            acc = self.eval(sess, x_, t_)
            logging.info('Epoch %d Training accuracy = %f, error rate = %f' % (epoch, acc, 1.0 - acc))

            if 1.0 - acc < self.tolerance:
                logging.info('Target error rate reached.')
                break
        
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
        K = self.n_classes
        
        x = np.asarray(x_)
        y = np.asarray(self.test(sess, x_))
        t = np.asarray(t_)

        # Calculate gradient
        grad = np.zeros([K, M])
        for j in xrange(K):
            for n in xrange(N):
                grad[j, :] += (y[n,j] - t[n,j]) * x[n]
        grad = -grad.flatten()
 
        # Calculate Hessian matrix
        I = np.identity(K)
        H = np.zeros([K*M, K*M])
        for j in xrange(K):
            for k in xrange(K):
                D_wjk = np.zeros([M, M])
                for n in xrange(N):
                    D_wjk += y[n,k] * (I[k,j] - y[n,j]) * x.T.dot(x)
                H[j*(M):(j+1)*M, (k)*M:(k+1)*M] = -D_wjk
        try:
            H_inv = np.linalg.pinv(H)
        except np.linalg.linalg.LinAlgError:
            return

        # Update weight
        w_old = self.w_k.flatten()
        w_new = w_old - self.lr * H_inv.dot(grad)
        w_new = w_new.reshape([K, M])
        self.w_k = w_new

class ProbabilisticGenerativeModel(object):
    def __init__(self, n_classes=1):
        self.classes = np.identity(n_classes)
        self.n_classes = n_classes

    def save(self, path):
        w = np.hstack((self.w0_k[:, np.newaxis], self.w_k))
        np.save(path, w)
        logging.info('Saving model to %s success [K = %d, M = %d]' % (path, w.shape[0], w.shape[1]))

    def load(self, path):
        w = np.load(path)
        self.w_k = w[:, 1:]
        self.w0_k = w[:, 0]
        self.classes = np.identity(w.shape[0])

        self.n_classes = w.shape[0]
        self.n_features = w.shape[1]
        logging.info('Loading model from %s success [K = %d, M = %d]' % (path, w.shape[0], w.shape[1]))
    
    def fit(self, sess, x_, t_, epoch=None, batch_size=None):
        assert len(x_.shape) == 2

        N, M = x_.shape
        K = len(self.classes)
        self.n_features = M
        self.w_k = np.zeros([K, M-1])
        self.w0_k = np.zeros([K])

        priors = np.zeros([K, 1])
        means = np.zeros([K, M-1])
        sigma = np.zeros([M, M-1])
    
        for class_i, i in zip(self.classes, range(K)):
            x_i = x_[t_[:, i] == 1][:, 1:]
            n_i = len(x_i)

            priors[i] = float(n_i) / N
            means[i] = np.mean(x_i, axis=0)
            sigma = (float(n_i) / N) * ((x_i - means[i]).T.dot(x_i - means[i])) / float(n_i)
        sigma_inv = np.asarray(np.linalg.pinv(sigma))
        
        for i in xrange(K):
            self.w_k[i, :] = (sigma_inv.dot(means[i]))
            self.w0_k[i] = (-1.0 / 2) * means[i].T.dot(sigma_inv.dot(means[i])) + np.log(priors[i])
   
    def test(self, sess, x_):
        N = len(x_)
        K = len(self.classes)
        a_k = np.zeros([N, K])
        x = np.asarray(x_[:, 1:])
        for k in xrange(K):
            a_k[:, k] = self.w_k[k, :].dot(x.T) + self.w0_k[k]
        return softmax(a_k)

    def eval(self, sess, x_, t_):
        y = self.test(sess, x_)
        t = np.asarray(t_)
        acc = float(np.equal(np.argmax(y, axis=1), np.argmax(t, axis=1)).sum()) / len(t) 
        return acc

