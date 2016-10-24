#!/usr/bin/env python
# -*-  coding: utf-8 -*-

from PIL import Image
from pylab import *
from scipy.misc import toimage
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1.定义直方图均衡化函数，这里传入的参数是灰度图像的数组和累积分布函数值
def histImageArr(im_arr, cdf):
    cdf_min = cdf[0]
    im_w = len(im_arr[0])
    im_h = len(im_arr)
    im_num = im_w * im_h
    color_list = []
    i = 0

    # 通过累积分布函数计算灰度转换值
    while i < 256:
        if i > len(cdf) - 1:
            color_list.append(color_list[i - 1])
            break
        tmp_v = (cdf[i] - cdf_min) * 255 / (im_num - cdf_min)
        color_list.append(tmp_v)
        i += 1

    # 产生均衡化后的图像数据
    arr_im_hist = []
    for itemL in im_arr:
        tmp_line = []
        for item_p in itemL:
            tmp_line.append(color_list[item_p])
        arr_im_hist.append(tmp_line)

    return arr_im_hist

# 封装一下图像处理的函数,cdf是累积分布函数数值
def beautyImage(im_arr):
    imhist, bins = np.histogram(im_arr.flatten(), range(256))
    cdf = imhist.cumsum()
    return histImageArr(im_arr, cdf)

def histEqual(srcpath, dstpath):
    im_source = Image.open(srcpath)

    if True:
        arr_im_gray = np.array(im_source)
        arr_im_gray_hist = beautyImage(arr_im_gray)
        # figure()
        im_conver = toimage(arr_im_gray_hist, 255, 0, None, None, None, 'L')
        im_conver.save(dstpath, 'jpeg')

# 2.二值化
def thresholdBinary(srcpath, dstpath):
    #  load a color image
    Lim = Image.open(srcpath)
    mtr = np.array(Lim)
    #  setup a converting table with constant threshold
    threshold = (mtr.max() + mtr.min())/2
    # threshold = 100
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    # convert to binary image by the table
    bim = Lim.point(table, '1')
    bim.save(dstpath, 'jpeg')


# 3.滤波
# 通用滤波
def generalBlur(srcpath, dstpath):
	img = cv2.imread(srcpath, 0) #直接读为灰度图像
	img1 = np.float32(img) #转化数值类型
	kernel = np.ones((5,5),np.float32)/25

	dst = cv2.filter2D(img1,-1,kernel)
	#cv2.filter2D(src,dst,kernel,auchor=(-1,-1))函数：
	#输出图像与输入图像大小相同
	#中间的数为-1，输出数值格式的相同plt.figure()
	plt.subplot(1,2,1), plt.imshow(img1,'gray')
	# plt.savefig('test1.jpg')
	plt.subplot(1,2,2), plt.imshow(dst,'gray')
	# plt.savefig('test2.jpg')
	plt.show()

# 均值滤波
def averageBlur(srcpath, dstpath):
	img = cv2.imread(srcpath, 0) #直接读为灰度图像
	blur = cv2.blur(img,(3,5))#模板大小3*5
	# cv2.imwrite(dstpath, blur)
	plt.subplot(1,2,1),plt.imshow(img,'gray')
	plt.subplot(1,2,2),plt.imshow(blur,'gray')
	plt.show()
	
# 高斯模糊
def gaussianBlur(srcpath, dstpath):
	img = cv2.imread(srcpath, 0) #直接读为灰度图像
	blur = cv2.GaussianBlur(img,(5,5),0)
	# cv2.imwrite(dstpath, blur)
	plt.subplot(1,2,1),plt.imshow(img,'gray')
	plt.subplot(1,2,2),plt.imshow(blur,'gray')
	plt.show()
	
# 中值滤波
def medianBlur(srcpath, dstpath):
	img = cv2.imread(srcpath, 0)
	blur = cv2.medianBlur(img, 3)
	# cv2.imshow(dstpath, img)
	# cv2.imwrite(dstpath, blur)
	plt.subplot(1,2,1),plt.imshow(img,'gray')
	plt.subplot(1,2,2),plt.imshow(blur,'gray')
	plt.show()
    
# 双边滤波
def bilateralFilter(srcpath, dstpath):
	img = cv2.imread(srcpath, 0)
	# 9---滤波领域直径
	# 后面两个数字：空间高斯函数标准差，灰度值相似性标准差
	blur = cv2.bilateralFilter(img,9,75,75)
	# cv2.imwrite(dstpath, blur)
	plt.subplot(1,2,1),plt.imshow(img,'gray')
	plt.subplot(1,2,2),plt.imshow(blur,'gray')
	plt.show()


if __name__ == '__main__':
    # histEqual('./original.jpg', './test.jpg')
    # thresholdBinary('./original.jpg', './test.jpg')
    # generalBlur('./test.png', './test.jpg')
    # averageBlur('./original.jpg', './test.jpg')
    # gaussianBlur('./original.jpg', './test.jpg')
    # medianBlur('./test.png', './test_out.jpg')
    bilateralFilter('./test.png', './test_out.jpg')

