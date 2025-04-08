# Celeb-Dataset-Analysis

## Overview

This project focuses on analyzing a celebrity dataset using advanced clustering techniques and embedding-based representations. The primary goal is to evaluate the clustering performance of doppelganger pairs using metrics such as V-Measure, Rand Score, and Silhouette Score. The analysis is conducted using KMeans and Agglomerative clustering algorithms, with visualizations and metrics generated for each pair.

The project demonstrates expertise in machine learning, data visualization, and embedding-based data analysis, making it a valuable addition to any portfolio.

---

## Key Features

1. **Clustering Analysis**:
   - Implements **KMeans** and **Agglomerative Clustering** to group embeddings.
   - Evaluates clustering performance using:
     - **V-Measure**: Assesses clustering quality based on homogeneity and completeness.
     - **Rand Score**: Measures similarity between clustering results.
     - **Silhouette Score**: Evaluates how well data points fit within their clusters.

2. **Embeddings Utilization**:
   - Leverages pre-trained embeddings to represent images in a high-dimensional space.
   - Processes embeddings to identify and analyze doppelganger pairs.

3. **Visualization**:
   - Generates detailed plots for clustering metrics.
   - Produces UMAP visualizations to explore the dataset in a reduced dimensional space.

4. **Batch Processing**:
   - Efficiently processes large datasets by batching clustering results and visualizations.

---

## Project Structure

### **1. clustering_Graphs.py**
This script is the core of the project, responsible for:
- Initializing embeddings using the `init_embeddings` function.
- Performing clustering using KMeans and Agglomerative Clustering.
- Calculating and plotting clustering metrics (V-Measure, Rand Score, Silhouette Score).
- Saving visualizations as PDF files for easy sharing and analysis.

Key constants:
- **EMBEDDINGS_PATH**: Path to the embeddings file.
- **EMBEDDINGS_IMAGE_PATH**: Path to the file containing image paths.
- **IMAGE_DIR**: Directory containing the images.
- **PAIRS_PATH**: CSV file containing doppelganger pairs.

### **2. z_Base_Dataset_Scores.ipynb**
- Provides an overview of clustering results for the dataset.
- Explores the performance of clustering algorithms using true labels.
- Highlights the effectiveness of each metric in evaluating clustering quality.

### **3. Clustering_Dataset_Score.ipynb**
- Generates CSV files containing clustering metrics for various numbers of clusters (2 to 100).
- Organizes results for both KMeans and Agglomerative Clustering.

### **4. umap.ipynb**
- Creates UMAP visualizations for the entire dataset and individual doppelganger pairs.
- Generates interactive HTML plots using Plotly, linking each data point to its corresponding image.

---

## Technical Details

- **Programming Language**: Python
- **Libraries Used**:
  - `pandas`, `numpy`: Data manipulation and numerical computations.
  - `matplotlib`, `plotly`: Data visualization.
  - `scikit-learn`: Clustering algorithms and metrics.
  - `tqdm`: Progress bar for batch processing.
  - `UMAP`: Dimensionality reduction for visualization.

- **Machine Learning Techniques**:
  - Clustering: KMeans, Agglomerative Clustering.
  - Evaluation Metrics: V-Measure, Rand Score, Silhouette Score.

- **Data Representation**:
  - Embeddings: High-dimensional representations of images.
  - UMAP: Non-linear dimensionality reduction for visualization.

---

## How to Run

1. **Setup**:
   - Ensure all dependencies are installed (`requirements.txt` can be created if needed).
   - Update the constants in `clustering_Graphs.py` to point to the correct file paths.

2. **Run Clustering Analysis**:
   - Execute `clustering_Graphs.py` to generate clustering metrics and visualizations.

3. **Explore Results**:
   - Review the generated plots in the `AGGLOMERATIVE` and `KMEANS` folders.
   - Use the Jupyter notebooks (`z_Base_Dataset_Scores.ipynb`, `Clustering_Dataset_Score.ipynb`, `umap.ipynb`) for further analysis and visualization.

---

## Applications

- **Data Science**: Demonstrates clustering and visualization techniques for high-dimensional data.
- **Machine Learning**: Highlights the use of embeddings and clustering metrics.
- **Portfolio Project**: Showcases expertise in Python, data analysis, and machine learning.

---

## Contact

For any questions or collaboration opportunities, feel free to reach out via [Email](nate.michelotti@gmail.com) or [LinkedIn](https://www.linkedin.com/in/nathan-michelotti/).
