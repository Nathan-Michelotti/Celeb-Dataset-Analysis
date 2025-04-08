Celeb-Dataset-Analysis

Overview

This project focuses on analyzing a celebrity dataset using advanced clustering techniques and embedding-based representations. The primary goal is to evaluate the clustering performance of doppelganger pairs using metrics such as V-Measure, Rand Score, and Silhouette Score. The analysis is conducted using KMeans and Agglomerative clustering algorithms, with visualizations and metrics generated for each pair.

The project demonstrates expertise in machine learning, data visualization, and embedding-based data analysis, making it a valuable addition to any portfolio.

Key Features

Clustering Analysis:

Implements KMeans and Agglomerative Clustering to group embeddings.

Evaluates clustering performance using:

V-Measure: Assesses clustering quality based on homogeneity and completeness.

Rand Score: Measures similarity between clustering results.

Silhouette Score: Evaluates how well data points fit within their clusters.

Embeddings Utilization:

Leverages pre-trained embeddings to represent images in a high-dimensional space.

Processes embeddings to identify and analyze doppelganger pairs.

Visualization:

Generates detailed plots for clustering metrics.

Produces UMAP visualizations to explore the dataset in a reduced dimensional space.

Distance-Based Analysis:

Computes inter-class center distances.

Ranks doppelganger similarity via proximity comparisons.

Project Structure

Celeb-Dataset-Analysis/
├── 0Hypersphere_Compairson/
│   ├── results/                      # Distance comparison CSVs
│   ├── Distances.ipynb              # Distance metrics + proximity ranking
│   ├── metrics.ipynb                # Per-pair clustering metric analysis
│   ├── test.ipynb                   # General testing and exploration
│   └── local_utils.py               # Embedding utilities and model logic
│
├── AGGLOMERATIVE/                   # Agglomerative clustering plots (PDF)
├── KMEANS/                          # KMeans clustering plots (PDF)
├── clusteringdata/
│   ├── Clustering_Dataset_Scores.py
│   ├── clustering_Graphs.py
│   └── Kmeans_Clustering_Per_Pair.csv
├── Umap_Comparisons/
│   ├── Combined_UMAP_Legend.pdf     # UMAP of full dataset
│   └── umap.py                      # UMAP generation (static + interactive)
├── requirements.txt
└── README.md

Technical Details

Language: Python

Libraries:

pandas, numpy: Data manipulation

matplotlib, plotly: Visualization

scikit-learn: Clustering & metrics

tqdm: Progress bars

umap-learn: Dimensionality reduction

torch, torchmetrics: Optional model logic & classification

Techniques:

Clustering: KMeans, Agglomerative

Evaluation: V-Measure, Rand Score, Silhouette Score

UMAP: Visualizing 512-dim embeddings in 2D

How to Run

Install Requirements

pip install -r requirements.txt

Update ConstantsEdit paths in:

clusteringdata/clustering_Graphs.py

clusteringdata/Clustering_Dataset_Scores.py

Umap_Comparisons/umap.py

Run Clustering Analysis

python clusteringdata/clustering_Graphs.py

Run Dataset-Level Evaluation

python clusteringdata/Clustering_Dataset_Scores.py

Generate UMAP Visualizations

python Umap_Comparisons/umap.py

Explore Additional Analysis
Open notebooks in 0Hypersphere_Compairson/ to dive deeper:

metrics.ipynb

Distances.ipynb

Applications

Data Science: Explore unsupervised clustering performance.

Machine Learning: Evaluate embeddings and distance-based identity similarity.

Portfolio Project: Demonstrates end-to-end ML pipeline from data ingestion to visualization.

Contact

For questions or collaboration opportunities, feel free to reach out:

📧 nate.michelotti@gmail.com🔗 LinkedIn – Nathan Michelotti

