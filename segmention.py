import os
from PIL import Image
import scipy.misc
import numpy as np

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imsave(images, path):
    scipy.misc.imsave(path, images)

def splitImage(srcpath):
    img = Image.open(srcpath)
    image = []
    # num = 0
    rowheight = 32
    colwidth = 32
    for r in xrange(8):
        for c in xrange(8):
            box = (c*rowheight, r*colwidth, (c+1)*rowheight, (r+1)*colwidth)
            image.append(np.array(img.crop(box)))
            # img.crop(box).save(os.path.join(dstpath, 'split_%02d.jpg'%(num,)), 'jpeg')
            # num += 1
    return image

'''
if __name__ == '__main__':
    splitImage('321.jpg','dcmx')
'''