import cv2
import numpy as np

import sys, os
from tqdm import *

from os import listdir
from os.path import isfile, join


'''
    Recursively load images in the directory and concate images together into a numpy array
    mypath: input directory
    out: output directory
'''
import logging

logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

mypath = sys.argv[1]
out = sys.argv[2]

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

logging.info('Convert images at %s to %s' % (mypath, out))
datas = []
for i in tqdm(range(len(onlyfiles))):
    img = cv2.cvtColor(cv2.imread(mypath + '/' + onlyfiles[i]), cv2.COLOR_RGB2GRAY)
    datas.append(img.flatten())
    
datas = np.asarray(datas)

np.save(out, datas)
