from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

outfile = './imagenet_file_list.txt'
flist_all = []

fpath='/export/home/hwang157/ILSVRC/Data/CLS-LOC/train'
for fp in listdir(fpath):
    if isfile(join(fpath, fp)):
        continue
        
    fpath2 = join(fpath, fp)
    flist_all += [join(fpath2, f) for f in listdir(fpath2) if isfile(join(fpath2, f))]

print('total number of images: {0}'.format(len(flist_all)))

with open(outfile, 'w') as fh:
    for ff in flist_all:
        fh.write("{0}\n".format(ff))