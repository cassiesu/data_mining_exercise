#!/usr/bin/env python2.7
import logging
import sys
import math
import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import euclidean_distances
import datetime

class ClusterCenter():
    def __init__(self, center):
        self.center = center
        self.points = []
        self.dist_sum = None
    def __len__(self):
        return len(self.points)
    def __getitem__(self, item):
        return self.points[item]
    def add_point(self, point):
        self.points.append(point)
        self.dist_sum = None
    def dist_point_sum(self):
        if self.dist_sum is None:
            self.dist_sum = 0.0
            for point in self:
                self.dist_sum += euclidean_distances(self.center, point, squared=True)
        return self.dist_sum
    def get_half_farthest_points(self):
        dist_points = [[euclidean_distances(self.center, p, squared=True), p] for p in self.points]
        sorted_dist_points = sorted(dist_points, key=lambda dist_point: dist_point[0])
        to_keep = len(self.points) / 2
        return [dist_point[1] for dist_point in sorted_dist_points][0:to_keep]

class DataPoint():
    def __init__(self, point, cluster):
        self.point = point
        self.cluster = cluster
        self.cluster.add_point(point)
        self.q = None
        self.dp_sum = None

    def calc_sampling_weight(self):
        if self.q is None:
            center_dist_ratio = 1.0
            if euclidean_distances(self.cluster.center, self.point, squared=True) != 0.0:
                center_dist_ratio = euclidean_distances(self.cluster.center, self.point, squared=True) / self.cluster.dist_point_sum()
            self.q = np.ceil(
                (5.0 / len(self.cluster)) +
                center_dist_ratio
            )
        return self.q

    def calc_sampling_probability(self):
        return self.calc_sampling_weight() / self.dp_sum

    def calc_weight(self, out_per_mapper):
        return 1.0 / self.calc_sampling_weight() / out_per_mapper

tweights=[];
tdata=[];

class Mapper:
    def __init__(self):
        self.no_clusters = 600
        self.out_per_mapper = 20000
        self.written = 0
        self.num_per_mapper = None
        self.cluster_centers = None
        self.cluster_center_points = None
        self.data_points = None
    def run(self):
        reader = sys.stdin
        arr = []
        for line in reader:
            arr.append(np.fromstring(line, dtype=np.float64, sep=' '))
        self.data = np.array(arr)
        self.num_per_mapper = len(self.data)
        np.random.shuffle(self.data)
        self.cluster_center_points = self.build_coresets()
        self.cluster_centers = [ClusterCenter(c) for c in self.cluster_center_points]
        self.sample_points()

    def write_feature(self, row, weight):
        tweights.append(np.float64(weight))
        tdata.append(row)
        self.written += 1

    def can_write_more_features(self):
        return self.written < self.out_per_mapper

    def cluster_center(self, cluster_index):
        return self.cluster_centers[cluster_index]

    def build_coresets(self):
        magic = 1000
        datalist = self.data.tolist()
        coreset = []
        while np.shape(datalist)[0]>0:
            coreset_part = datalist[0:magic]
            coreset += coreset_part
            datalist = np.delete(datalist, range(0, min(magic, len(datalist))), axis=0)
            if np.shape(datalist)[0]>0:
                datalist = self.remove_half_nearest_points(coreset_part, datalist)
        return coreset

    def remove_half_nearest_points(self, center_points, data):
        k = KMeans(n_clusters=self.no_clusters)
        k.cluster_centers_ = np.array(center_points)
        assigned_clusters = k.predict(np.array(data))
        clusters = [ClusterCenter(c) for c in center_points]
        for i in range(0, len(assigned_clusters)):
            clusters[assigned_clusters[i]].add_point(data[i])
        ret = []
        for c in clusters:
            ret += c.get_half_farthest_points()
        return ret

    def sample_points(self):
        k = KMeans(n_clusters=self.no_clusters)
        k.cluster_centers_ = np.array(self.cluster_center_points)
        assigned_clusters = k.predict(np.array(self.data))
        self.cluster_centers = [ClusterCenter(c) for c in self.cluster_center_points]
        self.data_points = [DataPoint(self.data[i], self.cluster_centers[assigned_clusters[i]]) for i in
                            range(len(self.data))]
        dp_sum = np.sum([dp.calc_sampling_weight() for dp in self.data_points]) / self.out_per_mapper
        for dp in self.data_points:
            dp.dp_sum = dp_sum
        while self.can_write_more_features():
            np.random.shuffle(self.data_points)
            for dp in self.data_points:
                if not self.can_write_more_features():
                    return
                dp.dp_sum = dp_sum
                if np.random.sample() < dp.calc_sampling_probability():
                    self.write_feature(dp.point, dp.calc_weight(self.out_per_mapper))


if __name__ == "__main__":
    m = Mapper()
    m.run()
    data = np.array(tdata)
    nSamples = data.shape[0]
    weights = np.array(tweights)
    indices = np.random.randint(nSamples, size=600)
    clusters = data[indices, :]

    km = KMeans(n_clusters=600, n_init=1)
    km.cluster_centers_ = clusters
    for t in range(500):
        cluster_indices = km.predict(data)
        for i in range(clusters.shape[0]):
            data_indices = np.where(cluster_indices == i)[0]
            weights_normalized = weights[data_indices] / np.float64(sum(weights[data_indices]))
            weights_normalized = np.reshape(weights_normalized, [data_indices.shape[0], 1])
            data_normalized = data[data_indices, :] * weights_normalized
            clusters[i, :] = np.sum(data_normalized, axis=0)
        km.cluster_centers_ = clusters

    def precise_str(x):
        return "%.25f" % x

    for r in range(clusters.shape[0]):
        print("%s" % " ".join(map(precise_str, clusters[r, :])))