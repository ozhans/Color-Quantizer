#!/usr/bin/env python
# -*- coding: utf-8 -*-


import copy
import numpy as np
class MyKMedoids:
    """KMedoids implementation parametric with 'pam' and 'clara' methods.

    Parameters
    ----------
    n_clusters : int, optional, default: 3
        The number of clusters to form as well as the number of medoids to
        determine.
    max_iter : int, default: 300
        Maximum number of iterations of the k-medoids algorithm for a
        single run.
    method : string, default: 'pam'
        If it is pam, it applies pam algorithm to whole dataset 'pam'.
        If it is 'clara' it selects number of samples with sample_ratio and applies
            pam algorithm to the samples. Returns best medoids of all trials
            according to cost function.
    sample_ratio: float, default: .2
        It is used if the method is 'clara'
    clara_trials: int, default: 10,
        It is used if the method is 'clara'
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples

    """

    def __init__(self, n_clusters=3, max_iter=300, method='clara', sample_ratio=.2, clara_trials=10, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.method = method
        self.sample_ratio = sample_ratio
        self.clara_trials = clara_trials
        self.random_state = np.random.RandomState(random_state)
        self.best_medoids = []
        self.min_cost = float('inf')
        self.data = []

    def fit(self, X):
        """Compute k-medoids clustering. If method is 'pam'
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        Returns
        ----------
        self : MyKMedoids
        """
        self.data = X
        if self.method == "pam":
            (m, c) = self.pam(X)
            self.best_medoids = m
            self.min_cost = c
        elif self.method == "clara":
            min_c = float('Inf')
            best_med = []
            for i in range(self.clara_trials):
                temp = self.sample()
                (m,c) = self.pam(temp)
                clusters = self.generate_clusters(m,X)  #2
                c = self.calculate_cost(m,clusters)
                if c < min_c:
                    min_c = c
                    best_med = m
            self.best_medoids = best_med
            self.min_cost = min_c
        return self

    def sample(self):
        """Samples from the data with given sample_ratio.

        Returns
        -------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        """
        return self.random_state.permutation(self.data)[:int(len(self.data)*self.sample_ratio)]


    def pam(self, X):
        """
        kMedoids - PAM
        See more : http://en.wikipedia.org/wiki/K-medoids
        The most common realisation of k-medoid clustering is the Partitioning Around Medoids (PAM) algorithm and is as follows:[2]
        1. Initialize: randomly select k of the n data points as the medoids
        2. Associate each data point to the closest medoid. ("closest" here is defined using any valid distance metric, most commonly Euclidean distance, Manhattan distance or Minkowski distance)
        3. For each medoid m
            For each non-medoid data point o
                Swap m and o and compute the total cost of the configuration
        4. Select the configuration with the lowest cost.
        5. repeat steps 2 to 4 until there is no change in the medoid.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        best_medoids, min_cost : tuple, shape [n_samples,]
            Best medoids found and the cost according to it.
        """
        temp = self.random_state.permutation(X)[:self.n_clusters]
        medoids = [temp[i].tolist() for i in range(len(temp))]

        result_medoids = medoids
        result_cost = float('Inf')
        temp_cost = float('Inf')
        prev_cost = result_cost
        clus = self.generate_clusters(medoids,X)
        for it in range(self.max_iter):
            for i in range(len(medoids)):
                for j in range(len(X)):
                    if X[j].tolist() not in result_medoids:
                        temp_medoids = copy.deepcopy(result_medoids)
                        temp_medoids[i] = X[j].tolist()
                        temp_cost = self.calculate_cost(temp_medoids,clus)

                    if temp_cost < result_cost:
                        result_cost = temp_cost
                        result_medoids = temp_medoids

            if prev_cost == result_cost:
                break;
            prev_cost = result_cost
        #print "Sample best medoids: ", result_medoids
        #print "Sample min cost: ", result_cost
        return (result_medoids,result_cost)

    def generate_clusters(self, medoids, samples):
        """Generates clusters according to distance to medoids. Order
        is same with given medoids array.
        Parameters
        ----------
        medoids: array_like, shape = [n_clusters, n_features]
        samples: array-like, shape = [n_samples, n_features]
        Returns
        -------
        clusters : array-like, shape = [n_clusters, elemens_inside_cluster, n_features]
        """
        clus = [[] for k in range(len(medoids))]
    
        for i in range(len(samples)):
            #val = distance.euclidean(samples[i],medoids[0])
            val = np.sum((samples[i]-medoids[0])**2)
            #val = np.linalg.norm(samples[i]-medoids[0])
            c = 0
            for j in range(len(medoids)):
                #v = np.linalg.norm(samples[i]-medoids[j])
                #v = distance.euclidean(samples[i],medoids[j])
                v = np.sum((samples[i]-medoids[j])**2)
                if v < val:
                    val = v
                    c = j
            clus[c].append(samples[i])
        return clus

    def calculate_cost(self, medoids, clusters):
        """Calculates cost of each medoid's cluster with squared euclidean function.
        Parameters
        ----------
        medoids: array_like, shape = [n_clusters, n_features]
        clusters: array-like, shape = [n_clusters, elemens_inside_cluster, n_features]
        Returns
        -------
        cost : float
            total cost of clusters
        """
        result = 0;
        for i in range(len(medoids)):
            for j in range(len(clusters[i])):
                #result += distance.euclidean(medoids[i],clusters[i][j])**2
                result += np.sum((medoids[i]-clusters[i][j])**2)
                #result += np.linalg.norm(medoids[i]-clusters[i][j])**2
        return result

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        result = []
        for i in range(X.shape[0]):
            min_i = 0
            #min_value = distance.euclidean(self.best_medoids[0],X[i])
            min_value = np.sum((self.best_medoids[0]-X[i])**2)
            #min_value = np.linalg.norm(self.best_medoids[0]-X[i])
            for j in range(1,len(self.best_medoids)):
                #temp = distance.euclidean(self.best_medoids[j],X[i])
                temp = np.sum((self.best_medoids[j]-X[i])**2)
                #temp = np.linalg.norm(self.best_medoids[j]-X[i])
                if temp < min_value:
                    min_i = j
                    min_value = temp
            result.append(min_i)
        res = np.array(result)
        return res

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        res = self.fit(X)
        return res.predict(X)


if __name__ == "__main__":
    X = np.array([np.array([2., 6.]),
                  np.array([3., 4.]),
                  np.array([3., 8.]),
                  np.array([4., 7.]),
                  np.array([6., 2.]),
                  np.array([6., 4.]),
                  np.array([7., 3.]),
                  np.array([7., 4.]),
                  np.array([8., 5.]),
                  np.array([7., 6.])

                  ])

    kmedoids = MyKMedoids(n_clusters=2, random_state=0,method="clara",sample_ratio=.5,clara_trials=10,max_iter=300)
    #print kmedoids.pam(X)
    print kmedoids.fit_predict(X)
    # [1 1 1 1 0 0 0 0 0 0]
    print kmedoids.best_medoids
    # [array([7., 4.]), array([2., 6.])]
    print kmedoids.min_cost
    # 28.0