import numpy
import algo
from parse import images, labels


BATCH_SIZE = 1
iter_count = len(images) // BATCH_SIZE


agent = algo.Agent()


for i in range(iter_count):
    sl = slice(i * BATCH_SIZE, i * BATCH_SIZE + BATCH_SIZE)
    batch = images[sl], labels[sl]
    agent.forward(batch)

    # Break early for testing (10000 iterations is a lot!)
    if i >= 100:
        break
