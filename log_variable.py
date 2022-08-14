import numpy as np 
import os 
import time 
from tqdm import tqdm 

H = [i + 1 for i in range(100)]
print(H)
Y = np.percentile(H, 100 -20)
print(Y)