import numpy as np
import cv2
from PIL import Image
import PIL
import os
import tflearn as tfl

def get_format(name):
	return name.rsplit('.')[-1]

def get_unFormat(name):
	return ''.join(name.rsplit('.')[:-1])

def distMap(frame1, frame2):
	"""outputs pythagorean distance between two frames"""
	frame1_32 = np.float32(frame1)
	frame2_32 = np.float32(frame2)
	diff32 = frame1_32 - frame2_32
	norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
	dist = np.uint8(norm32*255)
	mod = cv2.GaussianBlur(dist, (9,9), 0)
	return mod

def get_data_paths(folderPath, badAtr='_marked'):
	gudFormats = ['mp4', 'MP4', 'mkv', 'MKV', 'avi', 'AVI', 'log']

	vodFormats = ['mp4', 'MP4', 'mkv', 'MKV', 'avi', 'AVI']

	logFormats = ['log']

	pathList = [i[0] for i in os.walk(folderPath)]
	data_pathsX = []
	data_pathsY = []

	for i in pathList:
		for j in os.listdir(i):
			if os.path.isfile(i + '/' + j):
				if get_format(j) in gudFormats:
					if get_format(j) in vodFormats:
						data_pathsX.append(''.join([i, '/', j]))

					elif get_format(j) in logFormats:
						data_pathsY.append(''.join([i, '/', j]))

	useble_dataX = []

	for i in data_pathsX:
		nani = i.find(badAtr)
		if nani == -1:
			useble_dataX.append(i)

	data_paths = []

	for i in useble_dataX:
		data_paths.append([i, data_pathsY[data_pathsY.index(get_unFormat(i)+'.log')]])

	return data_paths

def paths_to_tensors(paths, ss=(80, 80)):
	print (paths)
	objsX = []
	objsY = []

	for i in paths:
		objsX.append(cv2.VideoCapture(i[0]))
		objsY.append(open(i[1], 'rt'))

	X = np.zeros((1, 80*80))
	Y = []

	for i in objsX:
		_, frame1 = i.read()

		X = np.concatenate((X, [np.zeros(80*80)]), axis=0)
		while 1:
			ret, frame = i.read()

			if ret != True:
				break

			imm = distMap(frame1, frame)

			frame1 = frame

			imm_pil = Image.fromarray(imm)
			imm_pil = imm_pil.resize(ss, resample=PIL.Image.BILINEAR)

			imm = np.reshape(np.array(imm_pil), 80*80)
			X = np.concatenate((X, [imm]), axis=0)

	for i in objsY:
		for j in i:
			Y.append(int(j))
	return X[1:], Y

def trim_to_size(x, y, maxlen=80):
	X = np.zeros((1, maxlen, x.shape[1]))
	Y = []

	while x.shape[0] > 0:
		if x[:80].shape[0] != 80:
			b = x[:80]
			kFactor = maxlen-b.shape[0]

			b = np.concatenate((b, np.zeros((kFactor, 80*80))), axis=0)

			X = np.concatenate((X, [b]), axis=0)
			break

		else:
			X = np.concatenate((X, [x[:80]]), axis=0)
			x = x[80:]

	while len(y) > 0:
		pad = y[:maxlen]
		if len(pad) != 80:
			for i in range(maxlen-len(pad)):
				pad.append(0)

			Y.append(pad)
			break
		else:
			Y.append(pad)
			y = y[maxlen:]

	return X[1:]/255, np.array(Y, dtype=np.float64)

##########gathering data###############

Xn, Yn = paths_to_tensors(get_data_paths('temp2_train'))

X, Y = trim_to_size(Xn, Yn)

########################################

print ('(`･ω･´)')

###########CutNN########################

cum = tfl.input_data([None, 80, 6400])
cum = tfl.lstm(cum, 10, dynamic=True, dropout=0.5, activation='relu')
cum = tfl.fully_connected(cum, 80, activation='softmax')

cum = tfl.regression(cum, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

lewd = tfl.DNN(cum, tensorboard_verbose=0)

lewd.load('CutNN')

########trainning#########################

lewd.fit(X, Y, n_epoch=90, shuffle=True, show_metric=True, batch_size=4, run_id='CutNN')

lewd.save('CutNN')

print (lewd.predict([X[0]]))

###########end#############################

print ('(*ゝω・)ﾉ')

