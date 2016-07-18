#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np
DIMENSION = 400  # Dimension of the original data.
w = []
for line in sys.stdin:
    line = line.strip()
    key, value = line.split("\t")
    w.append( eval(value) )
w = np.array(w).mean(axis=0)
print '\t'.join(map(str, w))
