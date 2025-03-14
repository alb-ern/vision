import numpy as np
import struct

image_filename = "t10k-images.idx3-ubyte"
label_filename = "t10k-labels.idx1-ubyte"
def load_mnist_images(filename):
	with open(filename, "rb") as f:
		magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
		data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
	return data


def load_mnist_labels(filename):
	with open(filename, "rb") as f:
		f.read(8)  # Skip header
		labels = np.frombuffer(f.read(), dtype=np.uint8)
	return labels

# Example usage:
images = load_mnist_images(image_filename)
labels = load_mnist_labels(label_filename)
