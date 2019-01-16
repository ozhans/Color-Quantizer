#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.ndimage as im
import scipy.misc as sm
import numpy as np
import PIL
from MyKMeans import MyKMeans

class ColorQuantizer:
    """Quantizer for color reduction in images. Use MyKMeans class that you implemented.
    
    Parameters
    ----------
    n_colors : int, optional, default: 64
        The number of colors that wanted to exist at the end.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Read more from:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    """
    
    def __init__(self, n_colors=64, random_state=None):
        self.image = None
        self.centers = None
        np.set_printoptions(threshold=np.nan)
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        


    def read_image(self, path):
        """Reads jpeg image from given path as numpy array. Stores it inside the
        class in the image variable.
        
        Parameters
        ----------
        path : string, path of the jpeg file
        """
        self.image = im.imread(path)
        self.d1 = self.image.shape[0]
        self.d2 = self.image.shape[1]
        self.d3 = self.image.shape[2]
        self.image = self.image.reshape((self.image.shape[0]*self.image.shape[1]),self.image.shape[2])

    def recreate_image(self, path_to_save):
        """Recreates image from the trained MyKMeans model and saves it to the
        given path.
        
        Parameters
        ----------
        path_to_save : string, path of the png image to save
        """
        self.image = self.image.reshape(self.d1,self.d2,self.d3)
        sm.imsave(path_to_save, self.image)
        pass

    def export_cluster_centers(self, path):
        """Exports cluster centers of the MyKMeans to given path.

        Parameters
        ----------
        path : string, path of the txt file
        """
        np.savetxt(path,self.kmeans.cluster_centers)
        
    def quantize_image(self, path, weigths_path, path_to_save):
        """Quantizes the given image to the number of colors given in the constructor.
        
        Parameters
        ----------
        path : string, path of the jpeg file
        weigths_path : string, path of txt file to export weights
        path_to_save : string, path of the output image file
        """
        """man_centers = np.zeros((64,3))
        for i in range(64):
            man_centers[i] = (255/64.0)*np.random.randint(64)
            print man_centers"""
        self.kmeans = MyKMeans(random_state=None,n_clusters=64,max_iter=600,init_method="random")
        
        self.read_image(path)
        self.centers = self.kmeans.initialize(self.image)
        #image_array_sample = shuffle(image_array, random_state=0)[:1000]
        temp_image = np.array(self.image)
        np.random.shuffle(temp_image)
        self.kmeans.fit(temp_image[:7500])
        labels = self.kmeans.predict(self.image)
        result = []
        cents = self.kmeans.cluster_centers
        for i in range(self.d1*self.d2):
            result.append(cents[labels[i]])
        self.image = np.array(result)   
        self.recreate_image(path_to_save)
        self.export_cluster_centers(weigths_path)
        


if __name__ == "__main__":
    if __name__ == "__main__":
        t = ColorQuantizer()
        t.quantize_image("../Docs/ankara.jpg","./ankara_cluster_centers.txt","./deneme-ankaradenemerand64.jpg")
