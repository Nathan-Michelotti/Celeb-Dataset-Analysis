# Celeb-Dataset-Analysis

# z_Base_Dataset_Scores.ipynb

This documentation is for files located in the `AGGLOMERATIVE` and `KMEANS` folders.

This code utilizes the embeddings dataset to map out data of the doppel pairs using true labels. The values listed below are plotted using KMeans and Agglomerative clustering with true labels, graphing the doppel pair for all the following metrics:

- **V_Measure**: This score is the mean between homogeneity and completeness scores, assessing the quality of clustering results. It indicates how tightly grouped the clusters are. The score ranges from 0 to 1, with 0 indicating poor clustering and 1 indicating perfect clustering.
  
- **Rand_Score**: This score measures the similarity between two data clusterings. It ranges from 0 to 1, with 0 indicating no connection between clusters and 1 showing that they are extremely similar.

- **Silhouette_Score**: This metric evaluates the effectiveness of a clustering technique. It measures how similar an object is to its own cluster compared to other clusters. Scores range from -1 to 1, where higher values indicate that each object is well matched to its respective cluster and poorly matched to neighboring clusters.

## Clustering_Dataset_Score.ipynb

**Folder**: `z_clusteringdata`

Using each respective clustering algorithm (KMeans, Agglomerative), CSVs have been created containing data for V Measure, Rand Score, and Silhouette Score, both per pair as well as per number of clusters from 2 to 100.

## umap.ipynb

- **UMAP**: Plots the total UMAP graph of the entire dataset, per doppelganger pair, as well as HTML links using Plotly to show the exact image each point is referring to.
