import numpy as np
h_l_bias = np.array(([1, 0.39558, 0.75548],
                [1, 0.47145, 0.58025],
                [1, 0.77841, 0.70603],
                [1, 0.50746, 0.71304]))
h_l = h_l_bias[:,1:]

ce = np.array(([-0.50135, 0.50135],
               [-0.50174, 0.50174],
               [0.49747, -0.49747],
               [0.49828, -0.49828]))

#print(np.multiply(temp.T, ce))

yhat=np.array(([0.49865, 0.50135],
               [0.49826, 0.50174],
               [0.49747, 0.50253],
               [0.49828, 0.50172]))
y = np.array(([1, 0],
              [1, 0],
              [0, 1],
              [0, 1]))
ce_grad = yhat - y
results = np.zeros((3,2))
for row in range(4):
    for weight in range(2):
        for node in range(3):
            results[node][weight] += ce_grad[node][weight] * h_l_bias[row][node]
print(results)
print(results/4)
#print(np.dot((h_l_bias).T, yhat-y))
#print(h_l_bias.T)
#print()
#print(yhat-y)
#print(yhat - y)
"""
From h0 to o0, x20 * (output 0 - target 0)
1 * (1 - 0.49865)
From h1 to o0,
0.39558 * (1 - 0.0.49826)

"""
