import glob
import os, sys
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt


def load_sound_files(files):
	sequences = []
	for f in files:
		X, sr = librosa.load(f)
		sequences.append((X, sr))
	print("Files loaded!")
	return sequences


def extract_features(f):
	X, sr = f
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sr).T, axis=0)
	return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(sequences):
	for i in range(len(sequences)):
		mfccs, chroma, mel, contrast, tonnetz = extract_features(sequences[i])
		sequence_features = np.hstack((mfccs, chroma, mel, contrast, tonnetz))
		sequence_features = np.expand_dims(sequence_features, axis=0)
		if i == 0:
			features = sequence_features
		else:
			features = np.concatenate((features, sequence_features))
		print("Audio sequence " + str(i+1) + "/" + str(len(sequences)) + " done!")
	return features


def main():
	film = sys.argv[1]

	files = []
	sub_path = [insert here the path to your audio files]
	i = 0
	while (os.path.exists(sub_path + str(i) + ".mp3")):
		files.append(sub_path + str(i) + ".mp3")
		i += 1

	sequences = load_sound_files(files)
	features = parse_audio_files(sequences)

	np.save("./data/audio/" + film, features)


main()
