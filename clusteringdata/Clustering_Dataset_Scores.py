import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, rand_score, v_measure_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from tqdm import tqdm

from local_utils import init_embeddings

# Constants for file paths
EMBEDDINGS_PATH = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/model_240000_DoppelVer_All_112x112_outputs.pth"
EMBEDDINGS_IMAGE_PATH = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/model_240000_DoppelVer_All_112x112_image_paths.txt"
IMAGE_DIR = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/DoppelVer_All_112x112"
DOPPELGANGER_PAIRS_PATH = "/home/nmichelotti/Desktop/Embeddings/Underlined_Pairs.csv"

def perform_clustering(X, y, range_clusters, base_path, method):
    """Performs clustering and logs results to a CSV file."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    results = []
    filename = os.path.join(base_path, f"{method}_Clustering_Per_Cluster.csv")
    file_exists = os.path.isfile(filename)

    for n_clusters in tqdm(range_clusters, desc=f"{method.capitalize()} Clustering"):
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)

        labels = clusterer.fit_predict(X_scaled)
        result = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_score(X_scaled, labels),
            'rand_index': rand_score(y, labels),
            'v_measure': v_measure_score(y, labels)
        }
        results.append(result)
        pd.DataFrame([result]).to_csv(filename, mode='a', header=not file_exists, index=False)
        file_exists = True

    return pd.DataFrame(results)

def perform_clustering_per_pair(pairs_csv, embeddings, base_path, method):
    """Performs clustering on each pair specified in a CSV file."""
    pairs_df = pd.read_csv(pairs_csv)
    results = []
    scaler = StandardScaler()
    umap_reducer = UMAP(n_components=2, random_state=42) if method == 'agglomerative' else None
    filename = os.path.join(base_path, f"{method}_Clustering_Per_Pair.csv")
    file_exists = os.path.isfile(filename)

    for index, row in tqdm(pairs_df.iterrows(), total=pairs_df.shape[0], desc="Processing Pairs"):
        name1, name2 = row['Pair 1'], row['Pair 2']
        indices_name1 = embeddings[embeddings['class'] == name1].index.tolist()
        indices_name2 = embeddings[embeddings['class'] == name2].index.tolist()

        if not indices_name1 or not indices_name2:
            continue

        combined_indices = indices_name1 + indices_name2
        X = embeddings.iloc[combined_indices, :-3]
        y_true = embeddings['class_num'].iloc[combined_indices].values

        X_scaled = scaler.fit_transform(X)
        if umap_reducer:
            X_scaled = umap_reducer.fit_transform(X_scaled)

        clusterer = KMeans(n_clusters=2, random_state=42) if method == 'kmeans' else AgglomerativeClustering(n_clusters=2)
        labels_pred = clusterer.fit_predict(X_scaled)
        result = {
            'Pairs': f"{name1} & {name2}",
            'n_clusters': 2,
            'silhouette_score': silhouette_score(X_scaled, labels_pred),
            'rand_index': rand_score(y_true, labels_pred),
            'v_measure': v_measure_score(y_true, labels_pred)
        }
        results.append(result)
        with open(filename, 'a') as f:
            header = 'Pairs,n_clusters,silhouette_score,rand_index,v_measure\n' if not file_exists else ''
            f.write(header + f"{result['Pairs']},{2},{result['silhouette_score']},{result['rand_index']},{result['v_measure']}\n")
        file_exists = True

    return pd.DataFrame(results)

def main():
    embeddings = init_embeddings(EMBEDDINGS_PATH, EMBEDDINGS_IMAGE_PATH, IMAGE_DIR)
    methods = ['agglomerative', 'kmeans']
    base_path = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8"

    for method in methods:
        method_path = os.path.join(base_path, method.upper())
        os.makedirs(method_path, exist_ok=True)
        # Perform general clustering
        general_clustering_results = perform_clustering(embeddings.iloc[:, :512], embeddings['class_num'], range(2, 100), method_path, method)

        # Perform clustering per pair
        pair_clustering_results = perform_clustering_per_pair(DOPPELGANGER_PAIRS_PATH, embeddings, method_path, method)

if __name__ == "__main__":
    main()
