import numpy as np
test = np.array([[1, 2, 3],[4,5,6],[7,8,9]])

permutation = np.random.permutation(3)
test2 = test[permutation]
print(permutation)
print(test2)
