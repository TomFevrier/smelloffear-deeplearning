def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, BatchNormalization
import numpy as np
import sys, os, argparse
import math
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('autoencoder_version',
					help="Autoencoder version")
parser.add_argument('model_version',
					help="Model version")
parser.add_argument('output',
					help="Output should be either 'labels' or 'vocs'.",
					choices=['labels', 'vocs'])
parser.add_argument('-y', '--y_test_structure',
					help="oso: One Screening Out\tor\tomo: One Movie Out\tor\tl20: Last 20%\t(if vocs)",
					choices=['oso', 'l20', 'omo'])
parser.add_argument('-bn',
					help="Batch normalization enabled",
					action='store_true')
args = parser.parse_args()

output = args.output

autoencoder_name = "./models/video_autoencoder_v" + args.autoencoder_version

if output == 'vocs':
	model_name = "./models/" + output + "_video_" +  args.y_test_structure + "_v" + args.model_version
else:
	model_name = "./models/" + output + "_video_v" + args.model_version

nb_classes = 13


def initiate_model():
	model = load_model(autoencoder_name)
	model.summary()
	print()

	if args.bn:
		nb_layers_to_delete = 8
	else:
		nb_layers_to_delete = 6

	for i in range(nb_layers_to_delete):
		model.pop()

	model.get_layer('conv3d_1').trainable = False
	model.get_layer('conv3d_2').trainable = False

	if args.bn:
		model.get_layer('batch_normalization_1').trainable = False
		model.get_layer('batch_normalization_2').trainable = False

	model.add(Flatten())

	if output == 'labels':
		model.add(Dense(1024, activation='relu'))
		model.add(BatchNormalization(name='batch_normalization_3'))
		model.add(Dense(nb_classes, activation='sigmoid'))

		model.compile(loss=keras.losses.binary_crossentropy,
					  optimizer=keras.optimizers.Adadelta())

	elif output == 'vocs':
		model.add(Dense(512, activation='relu'))
		model.add(BatchNormalization(name='batch_normalization_3'))
		model.add(Dense(6))

		model.compile(loss=keras.losses.mean_absolute_error,
					  optimizer=keras.optimizers.Adadelta())

	model.summary()
	print()

	return model


def main():
	model = initiate_model()
	model.save(model_name)


main()
