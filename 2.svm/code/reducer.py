#!/usr/bin/env python2.7
import sys
import numpy as np
import scipy
from sklearn.svm import LinearSVC


if __name__ == "__main__":
    reader = sys.stdin
    values = []
    for line in reader:
        key, w = line.split("\t")
        values.append(map(float, w.split(",")))

    npv = np.array(values)
    print " ".join(map(str, npv.mean(axis=0)))
