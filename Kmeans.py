__authors__ = ['1709992', '1711342', '1620854', '1641014']
__group__ = '13'

import numpy as np
import utils
from matplotlib import pyplot as plt

class KMeans:

    def __init__(self, X, K = 1, options = None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
        """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        self.X = X.astype(float)
        if self.X.ndim == 3:
            self.X = np.reshape(self.X, (-1, self.X.shape[2]) )

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        self.num_iter = 0
        self.centroids = np.zeros( (self.K, self.X.shape[1]) )
        self.old_centroids = np.zeros( (self.K, self.X.shape[1]) )
        if self.options['km_init'].lower() == 'random':
            self.centroids[:] = self.X[np.random.randint(self.X.shape[0], size = self.K), :]
        elif self.options['km_init'].lower() == 'custom':
            for k in range(self.K):
                self.centroids[k, :] = k * 255 / (self.K - 1)
        elif self.options['km_init'].lower() == 'first':
            i = 0
            self.centroids[0] = self.X[0, :]
            for k in range(1, self.K):
                i += 1
                while (self.centroids[:k] == self.X[i, :]).all(1).any(0):
                    i += 1
                self.centroids[k] = self.X[i, :]

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        distan = distance(self.X, self.centroids)
        self.labels = np.argmin(distan, 1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        np.copyto(self.old_centroids, self.centroids)
        for i in range(self.K):
            x = self.X[self.labels == i, :]
            if x.size > 0:
                self.centroids[i, :] = np.mean(x, 0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, atol = self.options['tolerance'], rtol = 0.0)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        self.get_labels()
        while not self.converges():
            self.num_iter += 1
            self.get_centroids()
            self.get_labels()

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        return ( ( (self.X - self.centroids[self.labels, :]) * (self.X - self.centroids[self.labels, :]) ).sum(axis = 1)).mean()

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        SSE = []
        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            WCD = self.withinClassDistance()
            print(WCD)
            SSE.append(WCD)

            if len(SSE) > 1 and SSE[-1] > SSE[-2] * 0.8:
                break

        self.K = self.K - 1
        self.fit()

        if self.options['verbose']:
            plt.subplot(133)
            plt.plot(SSE)
            plt.subplot(131)
            plt.axis('off')
            plt.imshow(self.X.reshape( (80, 60, 3) ) / 255)
            plt.subplot(132)
            plt.axis('off')
            plt.imshow( (self.centroids[self.labels] / 255).reshape( (80, 60, 3) ) )
            plt.show()

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    if X.ndim == 3:
        X = np.reshape(X, (-1, X.shape[-1]) )
    if C.ndim == 3:
        C = np.reshape(C, (-1, C.shape[-1]) )
    dist = np.zeros( (X.shape[0], C.shape[0]) )
    for i in range(C.shape[0]):
        dist[:, i] = np.sum( (X - C[i]) * (X - C[i]), 1)
    return np.sqrt(dist)

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    return list(map(lambda x: utils.colors[x], [np.argmax(i) for i in utils.get_color_prob(centroids)]))
