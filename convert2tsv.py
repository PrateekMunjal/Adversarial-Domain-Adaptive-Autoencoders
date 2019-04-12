import numpy as np

vectors = np.load('vectors.npy')
labels = np.load('labels.npy')
print(vectors.shape)
print(labels.shape)
np.savetxt('vectors.tsv', vectors, fmt="%i", delimiter="\t")

np.savetxt('labels.tsv', labels,fmt="%i",  delimiter="\t")