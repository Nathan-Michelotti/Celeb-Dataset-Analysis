import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import v_measure_score, rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from local_utils import init_embeddings

# Configurations
#warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
EMBEDDINGS_PATH = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/model_240000_DoppelVer_All_112x112_outputs.pth"
EMBEDDINGS_IMAGE_PATH = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/model_240000_DoppelVer_All_112x112_image_paths.txt"
IMAGE_DIR = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/DoppelVer_All_112x112"
PAIRS_PATH = "/home/nmichelotti/Desktop/Embeddings/Underlined_Pairs.csv"

embeddings = init_embeddings(EMBEDDINGS_PATH, EMBEDDINGS_IMAGE_PATH, IMAGE_DIR)

# Plotting functions
def plot_scores(sorted_scores, batch_index, folder_path, score_type):
    plt.figure(figsize=(15, 8))
    scores, labels = zip(*sorted_scores)  # Unzip scores and labels
    if score_type == 'silhouette':
        plt.scatter(range(len(scores)), scores, color='blue')
    else:
        plt.bar(range(len(scores)), scores, color='green' if score_type == 'v_measure' else 'blue')
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha='right')
    plt.ylabel(f'{score_type.replace("_", " ").title()} Score')
    plt.title(f'{score_type.replace("_", " ").title()} Scores Batch {batch_index}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=2)
    filename = f"{score_type}_Scores_Batch_{batch_index}.pdf"
    filepath = os.path.join(folder_path, filename)
    plt.savefig(filepath, format='pdf')
    plt.close()

# Calculation functions
def calculate_and_plot_scores(pairs_csv, folder_path, score_type, cluster_method):
    pairs_df = pd.read_csv(pairs_csv)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    all_scores_labels = []
    for index, row in tqdm(pairs_df.iterrows(), total=pairs_df.shape[0]):
        name1, name2 = row['Pair 1'], row['Pair 2']
        label = f"{name1} & {name2}"
        indices_name1 = embeddings.loc[embeddings['class'] == name1].index.tolist()
        indices_name2 = embeddings.loc[embeddings['class'] == name2].index.tolist()
        combined_indices = indices_name1 + indices_name2
        X = embeddings.iloc[combined_indices, :512]
        y_true = embeddings.loc[combined_indices, 'class_num'].values

        clusterer = KMeans(n_clusters=2, random_state=42) if cluster_method == 'kmeans' else AgglomerativeClustering(n_clusters=2)
        preds = clusterer.fit_predict(X)

        if score_type == 'v_measure':
            score = v_measure_score(y_true, preds)
        elif score_type == 'rand':
            score = rand_score(y_true, preds)
        else:  # silhouette
            score = silhouette_score(X, preds)

        plot_function = plot_scores if score_type == 'silhouette' else plot_scores
        all_scores_labels.append((score, label))

    all_scores_labels.sort(key=lambda x: x[0], reverse=True)
    batch_size = 40
    num_batches = len(all_scores_labels) // batch_size + (1 if len(all_scores_labels) % batch_size > 0 else 0)

    for i in range(num_batches):
        batch_scores_labels = all_scores_labels[i*batch_size:(i+1)*batch_size]
        plot_scores(batch_scores_labels, i+1, folder_path, score_type)

def main():
    clustering_methods = ['agglomerative', 'kmeans']
    score_types = ['v_measure', 'rand', 'silhouette']
    base_folder_paths = {
        'agglomerative': "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/AGGLOMERATIVE",
        'kmeans': "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/KMEANS"
    }

    for method in clustering_methods:
        for score_type in score_types:
            folder_path = os.path.join(base_folder_paths[method], f"{score_type.capitalize()}_Score_Comparisons")
            calculate_and_plot_scores(PAIRS_PATH, folder_path, score_type, method)

if __name__ == "__main__":
    main()
