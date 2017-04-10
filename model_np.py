import numpy as np
import logging
import random

def softmax(a):
    e = np.exp(a)
    return e / e.sum(axis=1)[:, np.newaxis]

class LinearClassificationModel(object):
    def __init__(self, optimizer, epochs=None, batch_size=None, tolerance=None):
        self.w = None
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance

    def save(self, path):
        np.save(path, self.w)
        logging.info('Saving model to %s success [K = %d, M = %d]' % (path, self.w.shape[0], self.w.shape[1]))

    def load(self, path):
        self.w = np.load(path)
        self.n_classes = self.w.shape[0]
        self.n_features = self.w.shape[1]
        logging.info('Loading model from %s success [K = %d, M = %d]' % (path, self.n_classes, self.n_features))
    
    def test(self, sess, x_):
        return softmax(x_.dot(self.w.T))

    def eval(self, sess, x_, t_):
        return float(np.equal(self.test(sess, x_).argmax(axis=1), t_.argmax(axis=1)).sum()) / len(t_)

    def fit(self, sess, x_, t_):
        self.n_classes = t_.shape[1]
        self.n_features = x_.shape[1]

        # Initialize weight for each class
        self.w = np.zeros([self.n_classes, self.n_features])

        if self.optimizer == 'seq':
            self._fit_sequential(sess, x_, t_, self.tolerance, self.epochs, self.batch_size)
        else:
            self._fit(sess, x_, t_)
    
    def _fit(self, sess, x_, t_):
        self._optimize(sess, x_, t_)
        acc = self.eval(sess, x_, t_)
        logging.info('Training accuracy = %f, error rate = %f' % (acc, 1.0 - acc))

    def _fit_sequential(self, sess, x_, t_, tolerance, epochs, batch_size):
        assert len(x_.shape) == 2

        N = len(x_)
        M = self.n_features
        K = self.n_classes

        # Iterative optimize
        indices = range(N)
        for epoch in xrange(epochs):
            np.random.shuffle(indices)
            for begin in xrange(0, N, batch_size):
                end = min(N, begin + batch_size)
                x_train = x_[indices[begin:end]]
                t_train = t_[indices[begin:end]]
                self._optimize(sess, x_train, t_train)
            acc = self.eval(sess, x_, t_)
            logging.info('Epoch %d Training accuracy = %f, error rate = %f' % (epoch, acc, 1.0 - acc))

            if 1.0 - acc < tolerance:
                logging.info('Target error rate reached.')
                break

    def _optimize(sess, x_, t_):
        raise NotImplementedError()         

class ProbabilisticDiscriminativeModel(LinearClassificationModel):
    def __init__(self, lr=0.01, epochs=20, batch_size=128, tolerance=0.01):
        super(ProbabilisticDiscriminativeModel, self).__init__('seq', epochs, batch_size, tolerance)
        self.lr = lr

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
            return

        # Update weight
        w_old = self.w.flatten()
        w_new = w_old - self.lr * H_inv.dot(grad)
        w_new = w_new.reshape([K, M])
        self.w = w_new

class ProbabilisticGenerativeModel(LinearClassificationModel):
    def __init__(self):
        super(ProbabilisticGenerativeModel, self).__init__(optimizer='once')

    def _optimize(self, sess, x_, t_):
        N, M = x_.shape
        K = self.n_classes

        priors = np.zeros([K, 1])
        means = np.zeros([K, M-1])
        sigma = np.zeros([M-1, M-1])
    
        for k in xrange(K):
            x_k = x_[t_[:, k] == 1][:, 1:]
            n_k = float(len(x_k))
            priors[k] = n_k / N
            means[k] = np.mean(x_k, axis=0)
            sigma += priors[k] * ((x_k - means[k]).T.dot(x_k - means[k])) / n_k
        sigma_inv = np.asarray(np.linalg.pinv(sigma))
        
        for k in xrange(K):
            self.w[k, 1:] = (sigma_inv.dot(means[k]))
            self.w[k, 0] = (-1.0 / 2) * means[k].T.dot(sigma_inv.dot(means[k])) + np.log(priors[k])
