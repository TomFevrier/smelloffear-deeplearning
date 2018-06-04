import cv2
import numpy as np
import sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('film',
					help="Name of the film")
parser.add_argument('nb_frames',
					help="Number of frames",
					type=int)
parser.add_argument('offset',
					help="Frame number to start with",
					type=int)
args = parser.parse_args()

film = args.film
nb_frames = args.nb_frames
offset = args.offset

width = 64
height = 36


def nb_frame(n):
	n += offset
	if n < 10:
		return "000" + str(n)
	if n < 100:
		return "00" + str(n)
	if n < 1000:
		return "0" + str(n)
	return str(n)

def main():
	sub_path = [insert here the path to your frames]
	for i in range(0, nb_frames//30):
		for j in range(30):
			img_loc = sub_path + film + "/" + nb_frame(i*30+j) + ".jpg"
			img = cv2.imread(img_loc)
			if (img is not None):
				img = np.reshape(img, (1, height, width, 3))
				if j == 0:
					sequence = img
				else:
					sequence = np.concatenate([sequence, img])
		sequence = np.reshape(sequence, (1, sequence.shape[0], height, width, 3))
		if i == 0:
			sequences = sequence
		else:
			sequences = np.concatenate([sequences, sequence])

	np.save("./data/pixels/" + film, sequences)


main()
