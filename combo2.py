import numpy as np
import cv2
from PIL import Image
import PIL
import os
import tflearn as tfl
from tflearn.data_utils import to_categorical, pad_sequences
import matplotlib.pyplot as plt
import tensorflow as tf

def distMap(frame1, frame2):
	"""outputs pythagorean distance between two frames"""
	frame1_32 = np.float32(frame1)
	frame2_32 = np.float32(frame2)
	diff32 = frame1_32 - frame2_32
	norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
	dist = np.uint8(norm32*255)
	mod = cv2.GaussianBlur(dist, (9,9), 0)
	return mod

def imToAr(imm):

	ss = (200, 200)

	img = Image.fromarray(imm)
	img = img.resize(ss, resample=PIL.Image.BILINEAR)
	
	img = np.array(img)

	img = img.astype(int)

	size = img.shape[0]*img.shape[1]

	img = np.reshape(img, size)

	return img/255.0

def imToSeq(path, maxlen):

	arr = np.array([np.zeros(200*200)])

	cum = cv2.VideoCapture(path)

	_, frame1 = cum.read()
	_, frame2 = cum.read()

	while cum.isOpened():
		ret, frame3 = cum.read()

		if ret != True:
			break

		dist = distMap(frame1, frame3)

		frame1 = frame2
		frame2 = frame3

		imm = imToAr(dist)

		arr = np.append(arr, [imm], axis=0)

		#cv2.imshow('hentai', frame)

		#k = cv2.waitKey(1)

		#if k%256 == 27:
		#	break

	cum.release()

	cv2.destroyAllWindows()

	if len(arr) > maxlen:
		arr = arr[:maxlen]


	elif len(arr) < maxlen:
		while len(arr) < maxlen:
			arr = np.append(arr, [np.zeros(200*200)], axis=0)


	return np.array(arr)

def get_format(str):
	dab = str.rsplit('.')

	return ''.join(dab[:-1]), dab[-1]

def get_data_paths(folderPath):
	gudFormats = ['mp4', 'MP4', 'mkv', 'MKV', 'avi', 'AVI']

	pathList = [i[0] for i in os.walk(folderPath)]
	data_pathsX = []

	for i in pathList:
		for j in os.listdir(i):
			if os.path.isfile(i + '/' + j):
				if get_format(j)[1] in gudFormats:
					data_pathsX.append(''.join([i, '/', j]))

	return data_pathsX

def lstm9ToWords(pre):
	undick = {'0':'no', '1':'whyNot', '2':'yes', '3':'taxi', '4':'noThx', '5':'night', '6':'niger', '7':'goodDay', '8':'notSign'}

	out = []

	oof = lambda a: a[0]

	h = 0

	while h < len(pre):
		c = []
		for i in zip(pre[h], range(len(pre[h]))):
			c.append(i)

		x = sorted(c, key=oof)[-1]

		print (x[0], undick[str(x[1])])
		out.append([undick[str(x[1])], x[0]])

		h += 1

	return out

################lstm9####################

n_clss = 9

nut = tfl.input_data([None, 80, 40000])
nut = tfl.lstm(nut, 20, dynamic=True, dropout=0.5, activation='relu')
nut = tfl.fully_connected(nut, n_clss, activation='softmax')
nut = tfl.regression(nut, optimizer='adam', learning_rate=0.0002, loss='categorical_crossentropy')

lewd2 = tfl.DNN(nut, tensorboard_verbose=0)

lewd2.load('lstm9_x2_res')

##########################################

print ('(☍﹏⁰)')

###########translating######################

paths = sorted(get_data_paths('temp'))

X = []

for path in paths:
	X.append(imToSeq(path, maxlen=80))

print ('(`･ω･´)')

pre = lewd2.predict(X)

tranlsated = lstm9ToWords(pre)

file = open('translated_raw.txt', 'wt')
file2 = open('translated.txt', 'wt')

for i in tranlsated:
	file.write('{0}\n'.format(i))

file.close()

normed_tran = []

old_i = None

for i in tranlsated:
	if i[0] != old_i and i[0] != 'notSign':
		normed_tran.append(i[0])
		old_i = i[0]

for i in normed_tran:
	file2.write('{0}\n'.format(i))

file2.close()

###############end########################
print ('(σ´-ω-`)σ')
