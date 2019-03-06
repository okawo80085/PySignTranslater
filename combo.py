import numpy as np
import cv2
from PIL import Image
import PIL
import os, shutil
import tflearn as tfl
from tflearn.data_utils import to_categorical, pad_sequences
import matplotlib.pyplot as plt
import tensorflow as tf

def imToAr(imm):

	ss = (200, 200)

	img = Image.fromarray(imm)
	img = img.resize(ss, resample=PIL.Image.BILINEAR)
	
	img = np.array(img)

	img = img.astype(int)

	size = img.shape[0]*img.shape[1]

	img = np.reshape(img, size)

	return img/255.0

def distMap(frame1, frame2):
	"""outputs pythagorean distance between two frames"""
	frame1_32 = np.float32(frame1)
	frame2_32 = np.float32(frame2)
	diff32 = frame1_32 - frame2_32
	norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
	dist = np.uint8(norm32*255)
	mod = cv2.GaussianBlur(dist, (9,9), 0)
	return mod

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

		imm = imToAr(imm)

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

def get_amount_of_cunts(path):
	log_file = open(path, 'rt')

	hole_thing = []

	for i in log_file:
		hole_thing.append(str(i[:-1]))
		#print (int(i[:-1]))
	log_file.close()

	X = ''.join(hole_thing).split('0')
	hole_thing = []

	for i in X:
		if len(i) > 0:
			hole_thing.append(i)

	return len(hole_thing)

def cut(path, logPath):
	name, exten = get_format(path)

	goodForms = ['mp4', 'MP4', 'mkv', 'MKV', 'avi', 'AVI']

	if exten in goodForms:
		print ('cutting')

	else:
		print ('bad path or file format')
		return False
	
	if os.path.isfile(path):
		cum = cv2.VideoCapture(path)
	else:
		print ('no such file or directory:', path)
		return False

	try:
		log_file = open(logPath, 'rt')
		getNumCutFrom = logPath

	except Exception as e:
		print ('can\'t open/find .log file, make sure it exists')
		print ('make sure that file: \'{0}\' exists'.format(logPath))
		print (e)
		return False

	hole_thing = []

	for i in log_file:
		hole_thing.append(int(i))
		#print (int(i[:-1]))
	log_file.close()

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')

	outs = []


	_, frame = cum.read()

	size = frame.shape
	size = (size[1], size[0])

	for i in range(get_amount_of_cunts(getNumCutFrom)):
		outs.append(cv2.VideoWriter('temp/edit{0}.mp4'.format(i),fourcc, 24.0, size, True))


	out_num = 0

	switch = 0

	h = 0

	print (size)

	while h < len(hole_thing):
		ret, frame = cum.read()

		if ret != True:
			break

		#print (hole_thing[h], frame.shape)

		if hole_thing[h] == 1 and switch == 0:
			switch = 1
			#print ('\t', out_num, switch)

		elif hole_thing[h] == 0 and switch == 1:
			out_num += 1
			switch = 0
			#print ('\t', out_num, switch)

		if switch == 1:
			recorder = outs[out_num]

			recorder.write(frame)

		h += 1

	for i in outs:
		i.release()

	cum.release()

	return True

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

			print (kFactor)

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

def norm(res, midval=0.0095):
	out = res > midval
	return np.array(out, dtype=np.float32)

def saveCut(arr, maxlen=-1):
	log = open('cut.log', 'wt')

	arr = arr[:maxlen]

	for i in arr[:-1]:
		log.write('{0}\n'.format(int(i)))

	log.write(str(int(arr[-1])))
	log.close()

	pass

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
		out.append(x)

		h += 1

	return out

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

def clean_temp(path):
	for the_file in os.listdir(path):
		file_path = os.path.join(path, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)

		except Exception as e:
			print(e)
			return False

	return True

def fake_mark_vod(path, form):
	log = open('{0}.{1}'.format(path, 'log'), 'wt')

	print ('{0}.{1}'.format(path, form))
	print ('{0}.{1}'.format(path, 'log'))

	cum = cv2.VideoCapture('{0}.{1}'.format(path, form))

	fn = 0

	while 1:
		ret, frame = cum.read()

		if ret != True:
			log.write('0')
			log.close()
			break

		if fn > 0:
			log.write('0\n')

		fn += 1

	cum.release()

	pass


###########cleanning temp##############

print ('cleanning temp')

if os.path.isdir('temp'):
	er = clean_temp('temp')

else:
	pp = os.getcwd() + '/temp'
	os.mkdir(pp)
	er = True

if er != True:
	print ('can\'t clean temp, clean it yourself')
	exit()

else:
	print ('temp cleanned')


#############CutNN######################

cum = tfl.input_data([None, 80, 6400])
cum = tfl.lstm(cum, 10, dynamic=True, dropout=0.5, activation='relu')
cum = tfl.fully_connected(cum, 80, activation='softmax')

cum = tfl.regression(cum, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

lewd = tfl.DNN(cum, tensorboard_verbose=0)

lewd.load('CutNN', weights_only=True)

#########################################

print ('(☍﹏⁰)')

'''
################lstm9####################

n_clss = 9

nut = tfl.input_data([None, 80, 40000])
nut = tfl.lstm(nut, 20, dynamic=True, dropout=0.5, activation='relu')
nut = tfl.fully_connected(nut, n_clss, activation='softmax')
nut = tfl.regression(nut, optimizer='adam', learning_rate=0.0002, loss='categorical_crossentropy')

lewd2 = tfl.DNN(nut, tensorboard_verbose=0)

tf.reset_default_graph()
lewd2.load('lstm9_x2_res', weights_only=True, create_new_session=False)

##########################################

print ('(☍﹏⁰)')
'''
##########Marcking for cut################

name, form = get_format('paht/to/your/data.mp4')

try:
	print ('marking video file')
	fake_mark_vod(name, form)
	print ('marked!')

except Exception as e:
	print ('can\'t mark the video file')
	exit()

Xn, Yn = paths_to_tensors([['{0}.{1}'.format(name, form), '{0}.log'.format(name)]])

X, Y = trim_to_size(Xn, Yn)

print ('(`･ω･´)')

pred = lewd.predict(X)

mark = np.reshape(pred, pred.shape[0]*pred.shape[1])

normed = norm(mark)

Y = np.reshape(Y, Y.shape[0]*Y.shape[1])

print (mark.shape, mark.dtype)
print (normed.shape, normed.dtype)
print (Y.shape, Y.dtype)

saveCut(normed, maxlen=len(Yn))

#tf.reset_default_graph()

################cutting#####################

er = cut('{0}.{1}'.format(name, form), 'cut.log')

if er != True:
	print ('something went wrong during cutting')
	exit()
'''
###########translating######################

paths = sorted(get_data_paths('temp'))

X = []

for path in paths:
	X.append(imToSeq(path, maxlen=80))

print ('(`･ω･´)')

pre = lewd2.predict(X)

tranlsated = lstm9ToWords(pre)

file = open('tranlsated.txt', 'wt')

for i in tranlsated:
	file.write('{0}\n'.format(i))

file.close()

###############end########################
'''
print ('(σ´-ω-`)σ')
