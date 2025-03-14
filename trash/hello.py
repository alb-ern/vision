import numpy as np
import tqdm

a=np.array([1,2])
b=np.array([[3,4,8],[5,6,98]])
print(a.dot(b))
print(a.shape)
m=np.array([1,2])
#print("ttt",a*b,a.T*b,a*b.T,a.T*b.T)
print(b.shape)


# def multi(a:np.ndarray,b:np.ndarray):
#     a2=np.array([a])
#     return (a2.T*b).sum(axis=1)