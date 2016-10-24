import os
from PIL import Image
import scipy.misc
import numpy as np

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imsave(images, path):
    scipy.misc.imsave(path, images)

def splitImage(srcpath,  dstpath):
    img = Image.open(srcpath)
    num = 0
    rowheight = 32
    colwidth = 32
    for r in xrange(8):
        for c in xrange(8):
            box = (c*rowheight, r*colwidth, (c+1)*rowheight, (r+1)*colwidth)
            print box
            img.crop(box).save(os.path.join(dstpath, 'split_%02d.jpg'%(num,)), 'jpeg')
            num += 1
if __name__ == '__main__':
    splitImage('321.jpg','dcmx')