import numpy as np
import imageio.v2 as imageio
import tqdm



size=200



arr = np.zeros((size,size), dtype=np.uint8)
for row in tqdm.tqdm(range(size)):
	for col in range(size):
		try:
			arr[row, col] = ((size/2-row)**3 + (size/2-col)**4 + np.random.randint(-50000,50000))/1000%255
		except:
			pass
	
imageio.imwrite("output.png", arr)
