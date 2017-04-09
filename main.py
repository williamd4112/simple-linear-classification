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
    X = []
    Y = []

    datasets = args.X.split(',')
    num_classes = len(datasets)

    for i in xrange(num_classes):
        x = np.matrix(np.load(datasets[i]))
        y = np.tile(one_hot(num_classes, i), [len(x), 1])
        X.append(x)
        Y.append(y)
        logging.info('Load %d data for class %d' % (len(x), i))
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    
    num_samples = len(X)
    inds = range(num_samples)
    np.random.shuffle(inds)

    n_train = int(args.frac * num_samples)
   
    X = X[inds]
    Y = Y[inds]
 
    logging.info('Preprocessing %d data...' % num_samples)
    X_normal = Preprocessor().normalize(X, INPUT_SPACE_RANGE)
    X_phi = Preprocessor().pca(X_normal, k=args.d)
    
    logging.info('Use model %s with %d-dim feautre space' % (args.model, args.d))
    sess = None
    model = get_model(args, num_classes)
    model.fit(sess, X_phi[:n_train], Y[:n_train], args.epoch, args.batch_size)

    logging.info('Evaluating...')
    evaluate(args, model, X_phi[n_train:], Y[n_train:])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train/test task', 
            choices=['train', 'test'], type=str, default='train')
    parser.add_argument('--X', help='data (ordered)', required=True, type=str)
    parser.add_argument('--model', help='gen/dis model', 
            choices=['gen', 'dis-newton', 'dis-sgd'], type=str, default='dis-newton')
    parser.add_argument('--d', help='pca dimension', type=int, default=2)
    parser.add_argument('--frac', help='fraction of training set', type=float, default=0.8)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('--epoch', help='epoch', type=int, default=1)
    parser.add_argument('--batch_size', help='batch size', type=int, default=2400)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    train(args)
