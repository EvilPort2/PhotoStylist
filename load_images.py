import cv2
import os
import glob
import numpy as np

CONTENT_DIR = "content/*.jpg"
STYLE_DIR = "style/*.jpg"

if os.name == 'nt':
	clear = 'cls'
else:
	clear = 'clear'

def load_content_image(shape=None):
	while True:
		os.system(clear)
		print("Select content image")
		print("--------------------")
		content_dir_files = glob.glob(CONTENT_DIR)
		for i, file in enumerate(glob.glob(CONTENT_DIR)):
			print("[%d]. %s" %(i, file))
		print('\n')
		
		choice = int(input('Enter your choice number: '))
		if choice < 0 or choice > len(content_dir_files)-1:
			input("Choice must be >= 0 and <=%d. Press enter to continue"%(len(content_dir_files)-1,))
			continue

		content_image = cv2.imread(content_dir_files[choice])
		content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
		content_image = content_image.astype(np.float64)
		if content_image.shape[0] * content_image.shape[1] > 500*500:  
			if shape != None:
				content_image = cv2.resize(content_image, (shape[1], shape[0]))
			else:
				content_image = cv2.resize(content_image, None, fx=0.5, fy=0.5)
		break
	return content_image

def load_style_image(shape=None):
	while True:
		os.system(clear)
		print("Select style image")
		print("------------------")
		style_dir_files = glob.glob(STYLE_DIR)
		for i, file in enumerate(glob.glob(STYLE_DIR)):
			print("[%d]. %s" %(i, file))
		print('\n')
		
		choice = int(input('Enter your choice number: '))
		if choice < 0 or choice > len(style_dir_files)-1:
			input("Choice must be >= 0 and <=%d. Press enter to continue"%(len(style_dir_files)-1,))
			continue

		style_image = cv2.imread(style_dir_files[choice])
		style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
		style_image = style_image.astype(np.float64)
		if shape != None:
			style_image = cv2.resize(style_image, (shape[1], shape[0]))
		else:
			style_image = cv2.resize(style_image, None, fx=0.5, fy=0.5)
		break

	return style_image
