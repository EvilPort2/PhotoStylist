from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Sequential
from load_images import load_content_image, load_style_image
import numpy as np
import cv2
from scipy.optimize import fmin_l_bfgs_b


def VGG16_AvgPool(shape):
	vgg16 = VGG16(input_shape=shape, weights='imagenet', include_top=False)
	model = Sequential()
	for layer in vgg16.layers:
		if layer.__class__ == MaxPooling2D:
			model.add(AveragePooling2D())
		else:
			model.add(layer)
	return model

def load_and_preprocess_content(shape):
	content = load_content_image(shape)
	content = np.expand_dims(content, axis=0)
	content = preprocess_input(content)
	return content

def load_and_preprocess_style(shape):
	style = load_style_image(shape)
	style = np.expand_dims(style, axis=0)
	style = preprocess_input(style)
	return style

def unpreprocess(img):
	img[..., 0] += 103.939
	img[..., 1] += 116.779
	img[..., 2] += 126.68
	return img

def scale_img(x):
	x = x - x.min()
	x = x / x.max()
	return x

def minimize_loss(fn, epochs, shape):
	img = np.random.randn(np.prod(shape))
	for i in range(epochs):
		img = np.reshape(img, newshape=(1, shape[0], shape[1], 3))
		temp = unpreprocess(img[0].copy())
		temp = scale_img(temp)
		cv2.imshow('img', temp)
		cv2.waitKey(2000)
		img, l, _ = fmin_l_bfgs_b(func=fn, x0=img, maxfun=20)
		img = np.clip(img, -127, 127)
		print("iteration = %d, loss = %f" %(i, l))
	return img