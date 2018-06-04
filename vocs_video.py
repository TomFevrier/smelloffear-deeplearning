def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
import numpy as np
import sys, os, argparse
import math
import h5py


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('model_version',
					help="Model version")
parser.add_argument('y_test_structure',
					help="oso: One Screening Out\tor\tomo: One Movie Out\tor\tl20: Last 20%",
					choices=['oso', 'l20', 'omo'])
parser.add_argument('-b', '--batch_size',
					help="Batch size",
					type=int,
					default=16)
parser.add_argument('-e', '--epochs',
					help="Number of epochs",
					type=int,
					default=10)
parser.add_argument('-f', '--film',
					help="Film to be left out for testing (if omo)")
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
y_test_structure = args.y_test_structure
film_tested = args.film

width = 64
height = 36

model_name = "./models/vocs_video_" + y_test_structure + "_v" + args.model_version

sequence_length = 30
input_shape = (sequence_length, height, width, 3)


class LossHistory(keras.callbacks.Callback):

	def __init__(self, films):
		self.films = films

	def on_train_begin(self, logs={}):
		self.losses = []

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		film_keys = list(self.films.keys())
		film_keys.sort()

		path = "./csv/performance_vocs_video_" + y_test_structure + "_v" + args.model_version + '".csv'"
		if not os.path.exists(path):
			fd = open(path, 'w+')
			fd.write("loss,val_loss,")
			for film in film_keys:
				fd.write(film + "_loss,")
			fd.write("\n")
			fd.close()

		row = str(round(logs.get('loss'), 4)) + "," + str(round(logs.get('val_loss'), 4)) + ","
		for film in film_keys:
			data = self.films[film]
			evaluation = self.model.evaluate(data[0], data[1])
			row += str(round(evaluation, 4)) + ","
		row += "\n"
		fd = open(path, 'a')
		fd.write(row)
		fd.close()


def initiate_model():
	model = load_model(model_name)
	model.summary()
	print()
	return model


def create_datasets():
	directory = "./data/vocs/"
	screenings_out = []
	films = {}

	count = 0

	for file in os.listdir(directory):
		film = file[file.find('_') + 1:-5]
		pixels = np.load("./data/pixels/" + film + ".npy")

		# C02 - Methanol - Ethanol - Acetone - Isoprene - Siloxane
		vocs = np.genfromtxt(directory + file, delimiter=',', skip_header=4756, usecols=(420, 452, 493, 523, 551, 848))

		voc_min = vocs.min(axis=(0, 1), keepdims=True)
		voc_max = vocs.max(axis=(0, 1), keepdims=True)
		vocs = (vocs - voc_min) / (voc_max - voc_min)

		indexes = np.where(np.isnan(vocs))
		vocs[indexes] = 0.5

		limit = min(pixels.shape[0], vocs.shape[0])

		if film not in films.keys():
			films[film] = [pixels[:limit], vocs[:limit]]
		else:
			films[film][0] = np.concatenate((films[film][0], pixels[:limit]))
			films[film][1] = np.concatenate((films[film][1], vocs[:limit]))

		if y_test_structure == 'l20':
			limit_training_set = math.floor(0.8 * limit)

			if file == os.listdir(directory)[0]:
				x_train = pixels[:limit_training_set]
				y_train = vocs[:limit_training_set]
				x_test = pixels[limit_training_set:limit]
				y_test = vocs[limit_training_set:limit]
			else:
				x_train = np.concatenate((x_train, pixels[:limit_training_set]))
				y_train = np.concatenate((y_train, vocs[:limit_training_set]))
				x_test = np.concatenate((x_test, pixels[limit_training_set:limit]))
				y_test = np.concatenate((y_test, vocs[limit_training_set:limit]))

			count += 1
			print("File " + str(count) + "/" + str(len(os.listdir(directory))) + " added to the datasets!")

		elif y_test_structure == 'oso':

			if film not in screenings_out:
				screenings_out.append(film)
				if 'x_test' not in locals():
					x_test = pixels[:limit]
					y_test = vocs[:limit]
				else:
					x_test = np.concatenate((x_test, pixels[:limit]))
					y_test = np.concatenate((y_test, vocs[:limit]))

			else:
				if 'x_train' not in locals():
					x_train = pixels[:limit]
					y_train = vocs[:limit]
				else:
					x_train = np.concatenate((x_train, pixels[:limit]))
					y_train = np.concatenate((y_train, vocs[:limit]))

			count += 1
			print("File " + str(count) + "/" + str(len(os.listdir(directory))) + " added to the datasets!")

	if y_test_structure == 'omo':

		for film, data in films.items():
			if film == film_tested:
				x_test = data[0]
				y_test = data[1]
			else:
				if 'x_train' not in locals():
					x_train = data[0]
					y_train = data[1]
				else:
					x_train = np.concatenate((x_train, data[0]))
					y_train = np.concatenate((y_train, data[1]))
			print(film + " added to the datasets!")

	return x_train, y_train, x_test, y_test, films


def main():
	model = initiate_model()

	x_train, y_train, x_test, y_test, films = create_datasets()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print()

	history = LossHistory(films)

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
			  shuffle=False,
	          validation_data=(x_test, y_test),
			  callbacks=[history])

	model.save(model_name)

main()
