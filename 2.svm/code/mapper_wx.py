#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
import numpy.random as npr
from sklearn import linear_model
DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.
M =10000  # nb of samples to take

npr.seed(0)
ww = npr.standard_cauchy(size=(M,400))
b = npr.uniform(0, 2*np.pi, size = M) 
def transform(x_original):
    gamma = 2
    return np.cos(ww.dot(x_original)*gamma+b)*np.sqrt(2.0/M)

if __name__ == "__main__":
    x = None
    clf = linear_model.SGDClassifier(fit_intercept=False, alpha=0.00001, loss='hinge')
    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        y = np.array([int(label)])
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.
        clf.partial_fit(x,y, classes=CLASSES)
    w = clf.coef_.flatten()
    print "%s\t%s" % ('1', str(list(w)))
