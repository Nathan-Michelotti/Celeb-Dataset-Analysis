import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import plotly.express as px
from tqdm import tqdm

from local_utils import init_embeddings

# Constants
EMBEDDINGS_PATH = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/model_240000_DoppelVer_All_112x112_outputs.pth"
EMBEDDINGS_IMAGE_PATH = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/model_240000_DoppelVer_All_112x112_image_paths.txt"
IMAGE_DIR = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/DoppelVer_All_112x112"
PAIRS_PATH = "/home/nmichelotti/Desktop/Embeddings/Underlined_Pairs.csv"
UMAP_FOLDER_PATH = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/Umap_Comparisons"
PLOTLY_UMAP_FOLDER = "/home/nmichelotti/Desktop/Embeddings/embeddings_for_n8/Umap_Comparisons/Plotly_Pairs"

# Load embeddings
embeddings = init_embeddings(EMBEDDINGS_PATH, EMBEDDINGS_IMAGE_PATH, IMAGE_DIR)
pairs_df = pd.read_csv(PAIRS_PATH)

def plot_umap_per_pair(embeddings, pairs_df, folder_path):
    """Plots UMAP for each pair in the pairs dataframe."""
    umap_reducer = UMAP(n_neighbors=5, min_dist=0.3, n_components=2, metric='cosine', random_state=42)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for index, row in tqdm(pairs_df.iterrows(), total=pairs_df.shape[0]):
        name1, name2 = row['Pair 1'], row['Pair 2']
        indices_name1 = embeddings[embeddings['class'] == name1].index.tolist()
        indices_name2 = embeddings[embeddings['class'] == name2].index.tolist()
        combined_embeddings = pd.concat([embeddings.iloc[indices_name1, :512], embeddings.iloc[indices_name2, :512]])
        umap_embeddings = umap_reducer.fit_transform(combined_embeddings)

        plt.figure(figsize=(12, 8))
        mid_point = len(umap_embeddings) // 2
        plt.scatter(umap_embeddings[:mid_point, 0], umap_embeddings[:mid_point, 1], label=name1)
        plt.scatter(umap_embeddings[mid_point:, 0], umap_embeddings[mid_point:, 1], label=name2)
        plt.title(f'UMAP Projection of {name1} & {name2}')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.legend()
        filename = f"{name1}_vs_{name2}_UMAP_plot.pdf"
        filepath = os.path.join(folder_path, filename)
        plt.savefig(filepath, format ='pdf')
        plt.close()

def plot_umap_all_classes(embeddings, pairs_df, folder_path):
    """Generates a UMAP visualization for all unique classes."""
    unique_classes = pd.unique(pairs_df[['Pair 1', 'Pair 2']].values.ravel('K'))
    scaler = StandardScaler()
    indices = [i for cls in unique_classes for i in embeddings[embeddings['class'] == cls].index]
    X = embeddings.iloc[indices, :512]
    X_scaled = scaler.fit_transform(X)
    umap_reducer = UMAP(n_neighbors=5, min_dist=0.3, n_components=2, metric='cosine', random_state=42)
    umap_embedding = umap_reducer.fit_transform(X_scaled)

    plt.figure(figsize=(12, 10))
    for cls in unique_classes:
        cls_points = umap_embedding[(np.array(embeddings.loc[indices, 'class']) == cls)]
        plt.scatter(cls_points[:, 0], cls_points[:, 1], label=cls)
    plt.title('UMAP Projection of All Pairs')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, "Combined_UMAP_Legend.pdf")
    plt.savefig(filepath, format ='pdf')
    plt.show()
    
def plot_umap_interactive(embeddings, pairs_df, folder_path):
    """Creates interactive UMAP plots using Plotly for each pair in the pairs dataframe."""
    umap_reducer = UMAP(n_neighbors=5, min_dist=0.3, n_components=2, metric='cosine', random_state=42)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for index, row in tqdm(pairs_df.iterrows(), total=pairs_df.shape[0]):
        name1, name2 = row['Pair 1'], row['Pair 2']
        indices_name1 = embeddings[embeddings['class'] == name1].index.tolist()
        indices_name2 = embeddings[embeddings['class'] == name2].index.tolist()
        combined_embeddings = pd.concat([embeddings.iloc[indices_name1, :512], embeddings.iloc[indices_name2, :512]])
        umap_embeddings = umap_reducer.fit_transform(combined_embeddings)

        df_plot = pd.DataFrame(umap_embeddings, columns=['UMAP-1', 'UMAP-2'])
        df_plot['label'] = [name1] * len(indices_name1) + [name2] * len(indices_name2)

        fig = px.scatter(df_plot, x='UMAP-1', y='UMAP-2', color='label', hover_name='label')
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(title=f'UMAP Projection of {name1} vs {name2}', xaxis_title='UMAP-1', yaxis_title='UMAP-2')

        filepath = os.path.join(folder_path, f"{name1}_vs_{name2}_UMAP_plot.html")
        fig.write_html(filepath)


if __name__ == "__main__":
    # Plot UMAP for each pair
    plot_umap_per_pair(embeddings, pairs_df, os.path.join(UMAP_FOLDER_PATH, 'Individual_Pairs'))
    
    # Plot UMAP for all unique classes
    plot_umap_all_classes(embeddings, pairs_df, UMAP_FOLDER_PATH)

    # Create interactive UMAP plots using Plotly
    plot_umap_interactive(embeddings, pairs_df, PLOTLY_UMAP_FOLDER)