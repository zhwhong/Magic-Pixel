import os
import scipy.misc
import tensorflow as tf
# import Image
from PIL import Image

def deal_image(input, x1, y1, x2, y2):
	fname = input.split('/')[-1]
	a = scipy.misc.imread(input)
	#crop x1, y1, x2, y2
	im = Image.open(input)
	box = (int(x1),int(y1),int(x2),int(y2))
	print box
	img = im.crop(box)
	inp = img.resize((32, 32))
	#img.save('subpixel/data/celebA/test/'+'xyz.jpg', inp)
	fname = x1+'-'+y1+'-'+x2+'-'+y2+'-'+fname
	tmp_path = 'tmp/'+fname
	inp.save(tmp_path,'png')
	#transfer 
	os.system('python main.py --is_single False -is_small True --file_name %s' % fname)
	#outpath = transfer(tmp_path)
	outpath="out_%s" % fname
	print outpath
	return outpath
