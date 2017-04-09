import argparse
import sys
import logging
import numpy as np

from util import one_hot
from preprocess import Preprocessor
from model_np import ProbabilisticGenerativeModel, LogisticRegressionModel

INPUT_SPACE_RANGE = 255.0

def get_model(args, num_classes):
    if args.model == 'gen':
        return ProbabilisticGenerativeModel(num_classes)
    elif args.model == 'dis-sgd':
        return LogisticRegressionModel(num_classes, optimizer='sgd', lr=args.lr)
    else:
        return LogisticRegressionModel(num_classes)

def evaluate(args, model, x_, t_):
    sess = None
    accuracy = model.eval(sess, x_, t_)
    logging.info('Accuracy = %f' % accuracy)

def train(args):
    FRAC = args.frac
    X = []
    Y = []
    
    # Load datasets class by class
    datasets = args.X.split(',')
    num_classes = len(datasets)
    parts = np.zeros([num_classes, 2], dtype=np.int32)
    count = 0
    for i in xrange(num_classes):
        x = np.matrix(np.load(datasets[i]))
        y = np.tile(one_hot(num_classes, i), [len(x), 1])
        parts[i, 0] = count
        parts[i, 1] = len(x)
        X.append(x)
        Y.append(y)
        count += len(x)
        logging.info('Load %d data for class %d' % (len(x), i))

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    num_samples = len(X)
 
    # Preprocess datasets
    logging.info('Preprocessing %d data...' % num_samples)
    X_normal = Preprocessor().normalize(X, INPUT_SPACE_RANGE)
    X_phi = Preprocessor().pca(X_normal, k=args.d)
    bias = np.ones(num_samples)[:, np.newaxis]
    X_phi = np.hstack((X_phi, bias))

    # Partitioning datasets
    logging.info('Partitioning datasets...')
    X_phi_Train = []
    Y_Train = []
    X_phi_Test = []
    Y_Test = []
    
    if args.permu == 'balance':
        for k in xrange(num_classes):
            begin = parts[k, 0]
            end = begin + parts[k, 1]
            inds = range(begin, end)
            np.random.shuffle(inds)
            nk = parts[k, 1]
            nk_train = int(FRAC * nk)
            
            X_phi_Train.append(X_phi[inds[:nk_train]])
            Y_Train.append(Y[inds[:nk_train]])
            X_phi_Test.append(X_phi[inds[nk_train:]])
            Y_Test.append(Y[inds[nk_train:]])

        X_phi_Train = np.concatenate(X_phi_Train)
        Y_Train = np.concatenate(Y_Train)    
        X_phi_Test = np.concatenate(X_phi_Test)    
        Y_Test = np.concatenate(Y_Test)
        
        # Shuffle training, testing set
        logging.info('Shuffling datasets...')
        n_train = len(X_phi_Train)
        n_test = len(X_phi_Test)
        inds_train = range(n_train)
        inds_test = range(n_test)
        np.random.shuffle(inds_train)
        np.random.shuffle(inds_test)

        X_phi_Train = X_phi_Train[inds_train]
        Y_Train = Y_Train[inds_train]
        X_phi_Test = X_phi_Test[inds_test]
        Y_Test = Y_Test[inds_test]
    else:
        n_train = int(float(num_samples) * FRAC)
        inds = range(num_samples)
        np.random.shuffle(inds)
        X_phi = X_phi[inds]
        Y = Y[inds]
        X_phi_Train = X_phi[:n_train]
        Y_Train = Y[:n_train]
        X_phi_Test = X_phi[n_train:]
        Y_Test = Y[n_train:]

    logging.info('Training/Testing = %d/%d' % (len(Y_Train), len(Y_Test)))
    for k in xrange(num_classes):
        logging.info('# class-%d = %d' % (k, Y_Train[Y_Train.argmax(axis=1) == k].shape[0]))

    logging.info('Use model %s with %d-dim feautre space' % (args.model, args.d))
    sess = None
    model = get_model(args, num_classes)
    model.fit(sess, X_phi_Train, Y_Train, args.epoch, args.batch_size)

    logging.info('Evaluating...')
    evaluate(args, model, X_phi_Test, Y_Test)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train/test task', 
            choices=['train', 'test'], type=str, default='train')
    parser.add_argument('--X', help='data (ordered)', required=True, type=str)
    parser.add_argument('--model', help='gen/dis model', 
            choices=['gen', 'dis-newton', 'dis-sgd'], type=str, default='dis-newton')
    parser.add_argument('--permu', help='train/test task', 
            choices=['unbalance', 'balance'], type=str, default='unbalance')
    parser.add_argument('--d', help='pca dimension', type=int, default=2)
    parser.add_argument('--frac', help='fraction of training set', type=float, default=0.8)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('--epoch', help='epoch', type=int, default=1)
    parser.add_argument('--batch_size', help='batch size', type=int, default=2400)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    train(args)
