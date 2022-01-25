#############################################################################    
############################## UTILITIES ####################################
#############################################################################

import pandas as pd
import functions as f
import glob
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy as sc
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
from scipy.spatial import distance
import sklearn
from sklearn.metrics import silhouette_score
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer

def save_pickle(element, path):
    with open(f"{path}", 'wb') as f:
        pickle.dump(element, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(f"{path}", 'rb',) as f:
        return pickle.load(f)

#############################################################################    
############################## K-MEANS ######################################
#############################################################################

def clustering(D, k, partition, n_iterations=200):
    
    for _ in range(n_iterations): 
        centroids = []
        for idx in range(k):
          ## replaces centroids with partition average 
            centroids.append(D[partition==idx].mean(axis=0))
            centroids = list(np.vstack(centroids))
            
      ## assigns each point to the centroid with minimum distance  
    partition = np.argmin(distance.cdist(D, centroids, 'euclidean'),axis=1)

    return partition,centroids

def first_partition(D,k,n_iterations):
    
    # Select a random index for obtain the random center for the first parition
    idx = np.random.randint(0,len(D), k) 

    # Step 1 random choice of the center 
    centroids = D[idx, :]

    # Step 2 calculate euclidean distance and take the minimum distance
    partition = np.argmin(distance.cdist(D, centroids, 'euclidean'),axis=1)

    return partition


def kmeans(D,k,n_iterations=100):
    
    D = D.to_numpy()
    partition = first_partition(D,k,n_iterations)

    # Step 4 repeate the step with an iterative function for find the optimal partition
    partition, centroids = clustering(D,k,partition,n_iterations)

    return partition, centroids
