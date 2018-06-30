import numpy as np
from keras.models import Model
from keras.layers import Conv2D
import keras.backend as K
from scipy.optimize import fmin_l_bfgs_b
import cv2
from utils import minimize_loss, load_and_preprocess_style, VGG16_AvgPool, unpreprocess, scale_img

def gram_matrix(img):
	X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
	G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
	return G

def style_loss(y, t):
	return K.mean(K.square(gram_matrix(y)-gram_matrix(t)))

def main():
	img = load_and_preprocess_style()

	shape = img.shape[1:]
	vgg = VGG16_AvgPool(shape)
	symbolic_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.__class__ == Conv2D]

	style_model = Model(vgg.input, symbolic_conv_outputs)
	style_outputs = [K.variable(y) for y in style_model.predict(img)]
	print(len(style_outputs))

	loss = 0
	for symbolic, actual in zip(symbolic_conv_outputs, style_outputs):
		loss += style_loss(symbolic[0], actual[0])

	gradients = K.gradients(loss, style_model.input)
	get_loss_grads = K.function(inputs=[style_model.input], outputs=[loss]+gradients)

	def get_loss_grads_wrapper(x):
		l, g = get_loss_grads([x.reshape(img.shape)])
		return l.astype(np.float64), g.flatten().astype(np.float64)

	img = minimize_loss(get_loss_grads_wrapper, 10, shape)
	img = np.reshape(img, newshape=(1, shape[0], shape[1], 3))
	img = unpreprocess(img[0].copy())
	img = scale_img(img)

	cv2.imshow('style', img)
	cv2.waitKey(0)