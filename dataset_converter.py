import cv2
import numpy as np

import sys, os
from tqdm import *

from os import listdir
from os.path import isfile, join

mypath = sys.argv[1]
out = sys.argv[2]

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

datas = []
for i in tqdm(range(len(onlyfiles))):
    img = cv2.cvtColor(cv2.imread(mypath + '/' + onlyfiles[i]), cv2.COLOR_RGB2GRAY)
    datas.append(img.flatten())
    
datas = np.asarray(datas)

np.save(out, datas)
