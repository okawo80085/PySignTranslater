import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='what? it\'s a marcker software, what did you expect... *dab*')

parser.add_argument('-dm', '--displayMode', choices=[0, 1], type=int, help='you can only choose 1 or 0, show movement graph 1, don\'t show movement graph 0 (default)')
parser.add_argument('-smf', '--startMF', type=float, help='starting value for moveFactor, default is 100')
parser.add_argument('-mmx', '--moveMax', type=float, help='value for moveMax, optional, default is 118')
parser.add_argument('-mmn', '--moveMin', type=float, help='value for moveMin, optional, default is 113')
parser.add_argument('-mm', '--manualMode', choices=[0, 1] , type=int, help='turn manual mode 1 on, 0 off, default is 0')


requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('-mp', '--markPath', type=str, help='path to a marcker image, has to be same resolution as video, relative path is ok', required=True)
requiredNamed.add_argument('-vp', '--vodPath', type=str, help='path and a name to a video to marck, has to be same resolution as marcker image, relative path is ok, also make sure it has _marked in the name before file extention', required=True)

requiredNamed2 = parser.add_argument_group('also required if "--manualMode" is 1')
requiredNamed2.add_argument('-ol', '--outLog', type=str, help='path and name of an output file')

parser.set_defaults(displayMode=0)
parser.set_defaults(manualMode=0)
parser.set_defaults(startMF=100)
parser.set_defaults(moveMax=118)
parser.set_defaults(moveMin=113)

args = parser.parse_args()

if vars(args)['vodPath'] == None:
	print ('argument --vodPath is required')
	exit()
elif vars(args)['markPath'] == None:
	print ('argument --markPath is required')
	exit()

if vars(args)['manualMode']:
	print ('you are in manual mode')
	if vars(args)['outLog'] == None:
		print ('argument --outLog is required')
		exit()

def meanError(gotY, shouldBeY):
	return np.mean([gotY, shouldBeY], dtype=np.float64)

def showGraf(arr):
	plt.plot(arr)
	plt.grid(True)
	plt.show()

	pass

def get_format(str):
	dab = str.rsplit('_marked.')

	return ''.join(dab[:-1])

name = vars(args)['vodPath']

try:
	cum = cv2.VideoCapture(name)
	marker = cv2.imread(vars(args)['markPath'])

except Exception as e:
	print ('can\'t open/read video or marcker file')
	print (e)
	exit()

if not vars(args)['manualMode']:
	log = open('{0}.log'.format(get_format(name)), 'wt')

else:
	print ('writing to path "{0}"'.format(vars(args)['outLog']))
	log = open(vars(args)['outLog'], 'wt')
#log.write('0\n')

rec = 0

delay = 0.2

fn = 0

data1 = []

moveFactor1 = vars(args)['startMF']

moveMax = vars(args)['moveMax']
moveMin = vars(args)['moveMin']

switch = 0

while True:
	ret, frame = cum.read()
	
	if ret != True:
		break

	if fn > 1:
		moveFactor1 = meanError(marker, frame)
	
	data1.append(moveFactor1)

	if rec%2 == 0:
		if fn > 0:
			log.write('0\n')
			switch = 0
	else:
		log.write('1\n')
		switch = 1

	if fn%100 == 0:
		print (fn, rec, moveFactor1)


	if moveMin < moveFactor1 < moveMax:
		print (fn, rec, round(moveFactor1, 2), '\t\t´-ω-`')
		rec += 1

	fn += 1
log.write(str(switch))
log.close()


cum.release()

if vars(args)['displayMode']:
	showGraf(data1)

print ('(σ´-ω-`)σ')