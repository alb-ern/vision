import parse
import numpy as np



images = parse.images
labels = parse.labels



def view():
	for i in images[0]:
		for e in i:
			print(int(bool(e)),end="")
		print()

print(images[0].flatten())