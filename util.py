import pandas as pd
import numpy as np
import random
import csv

import cv2

def one_hot(n, i):
    v = np.zeros(n)
    v[i] = 1
    return v


if __name__ == '__main__':
    import sys
    X, Y = load_categorical_dataset(sys.argv[1], sys.argv[2])
    print X.shape, Y.shape, Y[0]
