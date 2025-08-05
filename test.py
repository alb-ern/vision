import numpy as np


a=(np.array([[1,2],
			 [3,4],
			 [5,6]]))
b=np.array([[2,1],
			[1,2]])
d=np.array([1,2])
c=a.dot(b)
print(c)
print(c+d)