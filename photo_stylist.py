import numpy as np
from keras.models import Model
from keras.layers import Conv2D
import keras.backend as K
from scipy.optimize import fmin_l_bfgs_b
import cv2
from utils import minimize_loss, load_and_preprocess_style, load_and_preprocess_content, VGG16_AvgPool, unpreprocess, scale_img
from style_model import style_loss
from content_model import VGG16_AvgPool_CutOff
import time
import sys, os

if os.name == 'nt':
	clear = 'cls'
else:
	clear = 'clear'

content_image = load_and_preprocess_content(shape=None)
shape = content_image.shape[1:]
h, w = content_image.shape[1:3]
style_image = load_and_preprocess_style(shape=(h, w))

os.system(clear)
print('Parameters')
print('----------')
epochs = input('Enter number of epochs (default 20): ')
if epochs.strip() == '':
	epochs = 20
else:
	epochs = int(epochs)

conv_n = input('Enter number of conv layers to use. Must be between 1 and 13 both inclusive (default 10): ')
if conv_n.strip() == '':
	conv_n = 10
else:
	conv_n = int(conv_n)

weights = input('Enter weights (default [1,1,1,1,1,1,1,1,1,1,1,1,1]): ')
if weights.strip() == '':
	weights = [1]*13
else:
	weights = eval(weights)


vgg = VGG16_AvgPool(shape)
vgg.summary()
conv_layers = [0,1,2,4,5,7,8,9,11,12,13,15,16,17]
content_model = Model(vgg.input, vgg.layers[conv_layers[conv_n]].get_output_at(0))
content_target = K.variable(content_model.predict(content_image))

symbolic_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.__class__ == Conv2D]
style_model = Model(vgg.input, symbolic_conv_outputs)
style_outputs = [K.variable(y) for y in style_model.predict(style_image)]

loss = K.mean(K.square(content_target-content_model.output))
for w, symbolic, actual in zip(weights, symbolic_conv_outputs, style_outputs):
	loss += w*style_loss(symbolic[0], actual[0])

gradients = K.gradients(loss, vgg.input)
get_loss_grads = K.function(inputs=[vgg.input], outputs=[loss]+gradients)

def get_loss_grads_wrapper(x):
	l, g = get_loss_grads([x.reshape(content_image.shape)])
	return l.astype(np.float64), g.flatten().astype(np.float64)

final_img = minimize_loss(get_loss_grads_wrapper, epochs, shape)
final_img = np.reshape(final_img, newshape=(1, shape[0], shape[1], 3))
final_img = unpreprocess(final_img[0].copy())
#final_img = scale_img(final_img)

result = np.hstack((unpreprocess(content_image[0].copy()), unpreprocess(style_image[0].copy()), final_img))
cv2.imwrite('result/'+str(time.time())+'.jpg', result)

cv2.imwrite(str(time.time())+'.jpg', final_img)
keypress = cv2.waitKey(10000)
