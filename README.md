# PCA for dimensionality reduction combined with Kmeans
![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Goal
The Goal of this notebook is to apply a dimensionality reduction on a big dataset in order to remove noise and then to apply the kmenas algorithm to divide the songs in clusters (music genres) and try to understand the results using pivotal tables.
## Overwiev

1. Preparing the dataset
2. PCA
3. K-means
4. Pivotal tables

## Content

- __main.ipynb__ is the main notebook 
- __functions.py__ contains the kmeans functions
- __data__ is a folder which contains:
     - _pca.pickle_ the dataset with the PCA already applied
     - _tracks.pickle.zip_ azipped pickle file of the tracks dataset used for the pivotal table
