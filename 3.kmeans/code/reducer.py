import sys
import numpy as np
import sklearn.cluster as cls
np.set_printoptions(threshold = np.inf)

coreset = []
for line in sys.stdin:
    line = line.strip()
    point = np.fromstring(line, sep = ' ')
    coreset.append(point)
coreset = np.vstack(coreset)

kmeans = cls.KMeans(n_clusters = 100, init = 'k-means++', n_init = 12, max_iter = 300, tol = 0.0001, copy_x = True)
kmeans.fit(coreset)

for i in xrange(100):
	print " ".join(map(str, kmeans.cluster_centers_[i]))