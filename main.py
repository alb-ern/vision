import numpy
import algo
from parse import images,labels


BATCH_SIZE=20
iter_count=len(images)//BATCH_SIZE


agent=algo.Agent()


for i in range(iter_count):
	batch=images[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
	agent.forward(batch)





