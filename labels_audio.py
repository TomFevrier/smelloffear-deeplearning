def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
import sys, os, argparse
import math
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('model_version',
					help="Model version")
parser.add_argument('-b', '--batch_size',
					help="Batch size",
					type=int,
					default=64)
parser.add_argument('-e', '--epochs',
					help="Number of epochs",
					type=int,
					default=100)
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs

model_name = "./models/labels_audio_v" + args.model_version

input_dim = 193
nb_classes = 13

class ClassificationReport(Callback):

	def __init__(self, test_data):
		self.test_data = test_data

	def on_epoch_end(self, epoch, logs={}):
		predict = np.asarray(self.model.predict(self.test_data[0]).round())
		target = self.test_data[1]
		f1 = f1_score(target, predict, average='weighted')
		recall = recall_score(target, predict, average='weighted')
		precision = precision_score(target, predict, average='weighted')
		metrics = "-- f1: " + "{0:.3f}".format(f1) + " -- precision: " + "{0:.3f}".format(precision) + " -- recall " + "{0:.3f}".format(recall)
		print(metrics)

		path = "./csv/performance_labels_audio_v" + args.model_version + ".csv"
		if not os.path.exists(path):
			fd = open(path, 'w+')
			fd.write("f1,precision,recall\n")
			fd.close()
		row = str(round(f1, 4)) + "," + str(round(precision, 4)) + "," + str(round(recall, 4)) + "\n"
		fd = open(path, 'a')
		fd.write(row)
		fd.close()


def initiate_model():
	if not os.path.exists(model_name):
		model = Sequential()
		model.add(Dense(280, input_dim=input_dim))
		model.add(Dense(300, activation='tanh'))
		model.add(Dense(nb_classes, activation='sigmoid'))

		model.compile(loss=keras.losses.binary_crossentropy,
					  optimizer=keras.optimizers.Adadelta())

	else:
		model = load_model(model_name)

	model.summary()
	print()
	return model


def create_datasets():
	films = ['Buddy', 'Hobbit', 'Machete', 'Mitty', 'Paranormal', 'Tribute']

	for film in films:
		features = np.load("./data/audio/" + film + ".npy")

		useful_classes = (0, 4, 14, 15, 16, 17, 18, 21, 23, 24, 28, 33, 41)
		labels = np.genfromtxt("./data/labels/" + film + "_transposed.csv", delimiter=',', usecols=useful_classes)

		limit = min(features.shape[0], labels.shape[0])
		limit_training_set = math.floor(0.8 * limit)

		if film == films[0]:
			x_train = features[:limit_training_set]
			y_train = labels[:limit_training_set]
			x_test = features[limit_training_set:limit]
			y_test = labels[limit_training_set:limit]
		else:
			x_train = np.concatenate((x_train, features[:limit_training_set]))
			y_train = np.concatenate((y_train, labels[:limit_training_set]))
			x_test = np.concatenate((x_test, features[limit_training_set:limit]))
			y_test = np.concatenate((y_test, labels[limit_training_set:limit]))

	return x_train, y_train, x_test, y_test


def evaluate_model(y_test, y_pred):
	target_names = ['suspense', 'comedy', 'drama', 'everyday life', 'dream', 'landscape', 'conversation', 'conversation main actor', 'action', 'death', 'running', 'blood (violence)', 'sudden shock']

	print(classification_report(y_test, y_pred, target_names=target_names))


def main():
	model = initiate_model()

	x_train, y_train, x_test, y_test = create_datasets()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print()

	report = ClassificationReport((x_test, y_test))

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test),
			  callbacks=[report])

	y_pred = np.asarray(model.predict(x_test).round())
	evaluate_model(y_test, y_pred)

	model.save(model_name)


main()
