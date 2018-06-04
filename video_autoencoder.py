def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys, os, argparse
import math, random
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('model_version',
					help="Model version. If non-existent a new model is created.")
parser.add_argument('-b', '--batch_size',
					help="Batch size",
					type=int,
					default=16)
parser.add_argument('-e', '--epochs',
					help="Number of epochs",
					type=int,
					default=10)
parser.add_argument('-bn',
					help="Enable batch normalization",
					action='store_true')
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs

model_name = "./models/video_autoencoder_v" + args.model_version

width = 64
height = 36

sequence_length = 30
input_shape = (sequence_length, height, width, 3)


class LossHistory(Callback):

	def __init__(self, x_test):
		self.x_test = x_test

	def on_epoch_end(self, epoch, logs={}):
		path = './csv/performance_autoencoder_v' + args.model_version + '.csv'
		if not os.path.exists(path):
			fd = open(path, 'w+')
			fd.write("loss,val_loss\n")
			fd.close()
		row = str(round(logs.get('loss'), 4)) + "," + str(round(logs.get('val_loss'), 4)) + "\n"
		fd = open(path, 'a')
		fd.write(row)
		fd.close()
		save_reconstruction(self.model, self.x_test)


def initiate_model():
	if not os.path.exists(model_name):
		autoencoder = Sequential()
		autoencoder.add(Conv3D(32, kernel_size=3,
							   activation='relu',
							   padding='same',
							   input_shape=input_shape))
		if args.bn:
			autoencoder.add(BatchNormalization())
		autoencoder.add(MaxPooling3D(pool_size=2))
		autoencoder.add(Conv3D(64, kernel_size=3,
							   activation='relu',
							   padding='same'))
		if args.bn:
			autoencoder.add(BatchNormalization())
		autoencoder.add(MaxPooling3D(pool_size=2))
		autoencoder.add(Conv3D(64, kernel_size=3,
							   activation='relu',
							   padding='same'))
		if args.bn:
			autoencoder.add(BatchNormalization())
		autoencoder.add(UpSampling3D(size=2))
		autoencoder.add(Conv3D(32, kernel_size=3,
							   activation='relu',
							   padding='same'))
		if args.bn:
			autoencoder.add(BatchNormalization())
		autoencoder.add(UpSampling3D(size=2))
		autoencoder.add(ZeroPadding3D(padding=(1, 0, 0)))
		autoencoder.add(Conv3D(3, kernel_size=3,
							   activation='sigmoid',
							   padding='same'))

		autoencoder.compile(loss=keras.losses.binary_crossentropy,
					  		optimizer=keras.optimizers.Adadelta())

	else:
		autoencoder = load_model(model_name)

	autoencoder.summary()
	print()
	return autoencoder


def create_datasets():
	films = ['Buddy', 'Hobbit', 'Sauron', 'Machete', 'Machete_bis', 'Mitty', 'Hector', 'Paranormal', 'Paranormal_bis', 'Tribute', 'Tribute_bis', 'Furious']

	for film in films:
		pixels = np.load("./data/pixels/" + film + ".npy")
		print(film + ": " + str(pixels.shape[0]) + " sequences")

		limit = pixels.shape[0]
		limit_training_set = math.floor(0.8 * limit)

		if film == films[0]:
			x_train = pixels[:limit_training_set]
			x_test = pixels[limit_training_set:limit]
		else:
			x_train = np.concatenate((x_train, pixels[:limit_training_set]))
			x_test = np.concatenate((x_test, pixels[limit_training_set:limit]))

	return x_train, x_test


def save_reconstruction(model, x_test):
	decoded_imgs = model.predict(x_test)
	n = 12
	indexes = [17, 89, 142, 209, 249, 289, 333, 375, 404, 421, 502, 575]
	plt.figure(figsize=(20, 4))

	for i in range(n):
	    # Display original
	    ax = plt.subplot(2, n, i+1)
	    plt.imshow(cv2.cvtColor(x_test[indexes[i]][0], cv2.COLOR_BGR2RGB))
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)

	    # Display reconstruction
	    ax = plt.subplot(2, n, i + n + 1)
	    plt.imshow(cv2.cvtColor(decoded_imgs[indexes[i]][0], cv2.COLOR_BGR2RGB))
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	images = os.listdir('./autoencoder/')
	images.sort()
	last = images[-1]
	last = int(last[:2])
	plt.savefig("./autoencoder/" + str(last + 1).zfill(2) + "_epochs.png")


def main():

	autoencoder = initiate_model()

	x_train, x_test = create_datasets()
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	print()
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print()

	history = LossHistory(x_test)

	autoencoder.fit(x_train, x_train,
		            batch_size=batch_size,
		            epochs=epochs,
		            verbose=1,
				    shuffle=False,
		            validation_data=(x_test, x_test),
				    callbacks=[history])
	autoencoder.save(model_name)


main()
