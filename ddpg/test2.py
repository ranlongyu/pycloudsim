import numpy as np

l = 100
b = 5

indices = np.random.choice(range(6)[2:5], size=b)
indices2 = np.random.choice(l, size=b)
print(np.hstack([indices, indices2]))

print(list(range(0,8000,24)))