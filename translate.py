import argparse
import os, shutil
import subprocess as sub

############parsing arguments#############

parser = argparse.ArgumentParser(description='PySignTranslatEr, it\'s intended to be used for sign language translation, but it\'s undertrained, more details here https://github.com/okawo80085/PySignTranslater')

required = parser.add_argument_group('required arguments')

required.add_argument('-vp', '--videoPath', type=str, help='path to a video file to translate from', required=True)

args = parser.parse_args()

if not os.path.isfile(vars(args)['videoPath']):
	print ('has to be a file')
	exit()


################part 1####################

sub.run('python3 combo.py -vp {0}'.format(vars(args)['videoPath']), shell=True)

################part 2####################

sub.run('python3 combo2.py', shell=True)

###############end########################

print ('(っ・ω・）っ')
