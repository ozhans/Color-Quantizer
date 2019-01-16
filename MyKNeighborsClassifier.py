#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.spatial.distance as sp
import numpy as np
class MyKNeighborsClassifier:
    """Classifier implementing the k-nearest neighbors vote similar to sklearn 
    library but different.
    https://goo.gl/Cmji3U

    But still same.
    
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default.
    method : string, optional (default = 'classical')
        method for voting. Possible values:
        - 'classical' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'weighted' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - 'validity' weights are calculated with distance and multiplied
          of validity for each voter.  
        Note: implementing kd_tree is bonus.
    norm : {'l1', 'l2'}, optional (default = 'l2')
        Distance norm. 'l1' is manhattan distance. 'l2' is euclidean distance.
    Examples
    --------
    """
    def __init__(self, n_neighbors=5, method='classical', norm='l2'):
        self.n_neighbors = n_neighbors
        self.method = method
        self.norm = norm
        self.labels = []
        self.distinct_labels = []
        self.data = []
        self.validity = []

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Training data. 
        y : array-like, shape = [n_samples] 
            Target values.
        """
        self.data = X
        self.labels = y
        for i in range(len(self.labels)):
            if self.labels[i] not in self.distinct_labels:
                self.distinct_labels.append(self.labels[i])
        if self.method == "validity":
            result = np.zeros(len(X))
            for i in range(len(X)):
                val = np.zeros(len(self.distinct_labels))
                if self.norm == "l1":
                    distances = sp.cdist([X[i]],self.data,metric="cityblock")
                else:
                    distances = sp.cdist([X[i]],self.data)
                nearest_neighs = np.argpartition(distances,self.n_neighbors)
                for j in range(self.n_neighbors+1):
                    if nearest_neighs[0][j] != i:
                        val[self.labels[nearest_neighs[0][j]]] += 1/distances[0][nearest_neighs[0][j]]
                val = val/sum(val)
                result[i] = val[self.labels[i]]
            self.validity = result

            
    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Test samples.
        Returns
        -------
        y : array of shape [n_samples]
            Class labels for each data sample.
        """
        if len(self.labels) == 0:
            raise ValueError("You should fit first!")

        result = []
        if self.method == "classical":
            for i in range(len(X)):
                if self.norm == "l1":
                    distances = sp.cdist([X[i]],self.data,metric="cityblock")
                else:
                    distances = sp.cdist([X[i]],self.data)
                nearest_neighs = np.argpartition(distances,self.n_neighbors)
                calculated_label = np.zeros(len(self.distinct_labels))
                for j in range(self.n_neighbors):
                    ind = self.labels[nearest_neighs[0][j]]
                    calculated_label[ind] += 1
                result.append(np.argmax(calculated_label))
        elif self.method == "weighted":
            for i in range(len(X)):
                if self.norm == "l1":
                    distances = sp.cdist([X[i]],self.data,metric="cityblock")
                else:
                    distances = sp.cdist([X[i]],self.data)
                nearest_neighs = np.argpartition(distances,self.n_neighbors)
                calculated_label = np.zeros(len(self.distinct_labels))
                for j in range(self.n_neighbors):
                    ind = self.labels[nearest_neighs[0][j]]
                    calculated_label[ind] += 1/(distances[0][nearest_neighs[0][j]]+1e-15)
                result.append(np.argmax(calculated_label))
        elif self.method == "validity":
            for i in range(len(X)):
                if self.norm == "l1":
                    distances = sp.cdist([X[i]],self.data,metric="cityblock")
                else:
                    distances = sp.cdist([X[i]],self.data)
                nearest_neighs = np.argpartition(distances,self.n_neighbors)
                calculated_label = np.zeros(len(self.distinct_labels))
                for j in range(self.n_neighbors):
                    ind = self.labels[nearest_neighs[0][j]]
                    calculated_label[ind] += (1/(distances[0][nearest_neighs[0][j]]+1e-15))*self.validity[nearest_neighs[0][j]]
                result.append(np.argmax(calculated_label))
        return result
        
    def predict_proba(self, X, method=None):
        """Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Test samples.
        method : string, if None uses self.method.
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        result = []
        if method == "classical":
            for i in range(len(X)):
                if self.norm == "l1":
                    distances = sp.cdist([X[i]],self.data,metric="cityblock")
                else:
                    distances = sp.cdist([X[i]],self.data)
                nearest_neighs = np.argpartition(distances,self.n_neighbors)
                calculated_label = np.zeros(len(self.distinct_labels))
                for j in range(self.n_neighbors):
                    ind = self.labels[nearest_neighs[0][j]]
                    calculated_label[ind] += 1
                result.append(calculated_label/sum(calculated_label))
        elif method == "weighted":
            for i in range(len(X)):
                if self.norm == "l1":
                    distances = sp.cdist([X[i]],self.data,metric="cityblock")
                else:
                    distances = sp.cdist([X[i]],self.data)
                nearest_neighs = np.argpartition(distances,self.n_neighbors)
                calculated_label = np.zeros(len(self.distinct_labels))
                for j in range(self.n_neighbors):
                    ind = self.labels[nearest_neighs[0][j]]
                    calculated_label[ind] += 1/(distances[0][nearest_neighs[0][j]]+1e-15)
                result.append(calculated_label/sum(calculated_label))
        elif method == "validity":
            for i in range(len(X)):
                if self.norm == "l1":
                    distances = sp.cdist([X[i]],self.data,metric="cityblock")
                else:
                    distances = sp.cdist([X[i]],self.data)
                nearest_neighs = np.argpartition(distances,self.n_neighbors)
                calculated_label = np.zeros(len(self.distinct_labels))
                for j in range(self.n_neighbors):
                    ind = self.labels[nearest_neighs[0][j]]
                    calculated_label[ind] += (1/(distances[0][nearest_neighs[0][j]]+1e-15))*self.validity[nearest_neighs[0][j]]
                result.append(calculated_label/sum(calculated_label))
        return result

if __name__=='__main__':
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = MyKNeighborsClassifier(n_neighbors=3, method="validity")
    neigh.fit(X, y)
    #print neigh.predict(X)
    n = 0.9
    print(neigh.predict_proba([[n]], method='classical'))
    # [[0.66666667 0.33333333]]
    print(neigh.predict_proba([[n]], method='weighted'))
    # [[0.92436975 0.07563025]]
    print(neigh.predict_proba([[n]], method='validity'))
    # [[0.92682927 0.07317073]]