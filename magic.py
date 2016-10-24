import scipy.misc
import tensorflow as tf
import Image
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
	tmp_path = 'tmp/'+fname
	inp.save(tmp_path)
	#transfer 
	#outpath = transfer(tmp_path)
	outpath="out.jpg"
	return outpath
