from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

data, target = load_boston().data, load_boston().target
nor_data = (data - data.mean(axis = 0))/data.std(axis = 0)

U, s, V = np.linalg.svd(nor_data)
c1 = V.T[:, 0]
c2 = V.T[:, 1]
first_principal_components = nor_data.dot(c1)
second_principal_components = nor_data.dot(c2)

plt.scatter(first_principal_components, second_principal_components,
            c=target / max(target))