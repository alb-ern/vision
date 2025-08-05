import numpy as np

class Agent:
	def __init__(self) -> None:
		self.w=np.random.random((784,784))
		self.b=np.random.random(784)
	def forward(self,images):
		images = images.reshape(len(images), -1)
		return images
