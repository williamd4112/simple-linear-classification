import numpy as np
import csv
import sys

with open('answer_train.csv', 'w') as f:
    K = 3
    N = 1000
    I = np.identity(K).astype(np.int32)
    for k in xrange(K):
        yn = I[k, :]
        for i in xrange(N):
            f.write(','.join([str(yn_i) for yn_i in yn]) + '\n')
