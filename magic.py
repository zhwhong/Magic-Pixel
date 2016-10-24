import os
import scipy.misc
import tensorflow as tf
# import Image
from PIL import Image

def deal_image(input, x1, y1, x2, y2,size):
	fname = input.split('/')[-1]
	a = scipy.misc.imread(input)
	#crop x1, y1, x2, y2
	im = Image.open(input)
	box = (int(x1),int(y1),int(x2),int(y2))
	print box
	img = im.crop(box)
	#inp = img.resize((32, 32))
	inp = img.resize((int(size), int(size)))
	#img.save('subpixel/data/celebA/test/'+'xyz.jpg', inp)
	fname = x1+'-'+y1+'-'+x2+'-'+y2+'-'+fname
	tmp_path = 'tmp/'+fname
	inp.save(tmp_path)
	#transfer 
	if size == 128:
		os.system('python main.py --is_single False --is_small False --file_name %s' % fname)
	else :
		os.system('python main.py --is_single True  --file_name %s' % fname)
	os.system('rm tmp/*')
	#outpath = transfer(tmp_path)
	outpath="out_%s" % fname
	print outpath
	return outpath
