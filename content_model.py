from keras.layers import MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Sequential
import numpy as np
import keras.backend as K
import cv2
from utils import VGG16_AvgPool, load_and_preprocess_content, minimize_loss, unpreprocess, scale_img

def VGG16_AvgPool_CutOff(shape, num_conv):
	if num_conv < 1 or num_conv > 13:
		return None
	model = VGG16_AvgPool(shape)
	n = 0
	new_model = Sequential()
	for layer in model.layers:
		if layer.__class__ == Conv2D:
			n+=1
		new_model.add(layer)
		if n >= num_conv:
			break
	del model
	return new_model

def main():
	img = load_and_preprocess_content()
	shape = img.shape[1:]
	content_model = VGG16_AvgPool_CutOff(shape, 10)
	target = K.variable(content_model.predict(img))

	mean_squared_loss = K.mean(K.square(target-content_model.output))
	gradients = K.gradients(mean_squared_loss, content_model.input)
	get_loss_grads = K.function(inputs=[content_model.input], outputs=[mean_squared_loss]+gradients)

	def get_loss_grads_wrapper(x):
		l, g = get_loss_grads([x.reshape(img.shape)])
		return l.astype(np.float64), g.flatten().astype(np.float64)

	img = minimize_loss(get_loss_grads_wrapper, 10, shape)
	final_img = img.reshape(*shape)
	final_img = unpreprocess(final_img)
	final_img = scale_img(final_img)
	cv2.imshow('final_img', final_img)
	cv2.waitKey(0)
