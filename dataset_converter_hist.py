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

mypath = sys.argv[1]
out = sys.argv[2]

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

datas = []
for i in tqdm(range(len(onlyfiles))):
    img = cv2.cvtColor(cv2.imread(mypath + '/' + onlyfiles[i]), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    datas.append(np.squeeze(hist))
    
datas = np.asarray(datas)
print datas.shape

np.save(out, datas)
