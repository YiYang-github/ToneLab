import os
import umap
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE, MDS, Isomap

def match_transcription(feat, unique_transcription):
    """
    Matches the given feature vector to the closest row in the unique transcription tensor.

    Args:
    - feat: torch.tensor of shape (3), the feature vector to match.
    - unique_transcription: torch.tensor of shape (37, 3), the tensor containing unique transcriptions.

    Returns:
    - torch.tensor of shape (3), the closest matching row from unique_transcription.
    """

    # Compute the absolute difference between feat and each row in unique_transcription
    diff = torch.abs(unique_transcription - feat)

    # Sum the absolute differences along the second dimension to get a single value for each row
    diff_sum = diff.sum(dim=1)

    # Find the index of the row with the smallest sum of differences
    min_index = torch.argmin(diff_sum)

    # Return the row from unique_transcription that has the smallest difference
    return unique_transcription[min_index]

def reduce_dim(matrix: np.ndarray, dim: int = 2, method: str = 'pca') -> np.ndarray:
    """
    Reduce the dimensionality of the given matrix using the specified method.

    Args:
    - matrix (np.ndarray): Input matrix with shape [n_samples, n_features]
    - dim (int): Target dimensionality
    - method (str): Dimensionality reduction method ('pca', 'umap', 'tsne', 'mds', 'isomap')

    Returns:
    - np.ndarray: Reduced dimension matrix with shape [n_samples, dim]
    """
    if method == 'pca':
        reducer = PCA(n_components=dim)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=dim)
    elif method == 'tsne':
        reducer = TSNE(n_components=dim)
    elif method == 'mds':
        reducer = MDS(n_components=dim, dissimilarity='euclidean')
    elif method == 'isomap':
        reducer = Isomap(n_components=dim)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'pca', 'umap', 'tsne', 'mds', 'isomap'.")

    reduced_matrix = reducer.fit_transform(matrix)
    return reduced_matrix

def auto_cluster_feat(X, checkpoint_dir, layer_index, unique_transcription, device):
    """
    Extract features from the intermediate layers of a trained transcription model and normalize them.

    Args:
        X (torch.Tensor): Input features of shape (num_samples, feature_shape).
        checkpoint_dir (str): Directory containing the model checkpoint.
        layer_index (int): The index of the intermediate layer to extract features from.
        unique_transcription (list): List of unique transcriptions for decoding.
        device (str): Device to run the model on.

    Returns:
        tuple: Normalized embeddings and predicted sequences.
    """
    model = model.DenseNet(k=X.shape[1], pretrained=True, normalize_values=False, classes=3).to(device)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    model_state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(model_state_dict)

    model.eval()
    with torch.no_grad():
        feats = X.to(device)
        values = model(feats)
        pred_seq = [match_transcription(values[index].cpu(), unique_transcription) for index in range(values.shape[0])]

        # Get the embedding
        if layer_index == -1:
            embed = model.get_last_layer_features(feats).cpu().numpy()
        else:
            embed = model.get_features_by_index(feats, layer_index).cpu().numpy()

        # Normalize the embeddings
        embed = normalize(embed)
    return embed, pred_seq

def auto_cluster_plot(embed, pred_seq, eps=0.5, min_samples=2, type_='raw'):
    """
    Clusters and visualizes the embeddings using DBSCAN.

    Args:
        embed (np.ndarray): Normalized embeddings.
        pred_seq (list): Predicted sequences.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        type_ (str): Type of clustering to perform ('raw' or 'umap').

    Returns:
        dict: Unique class names for each cluster.
    """
    # Reduce the dimensionality for visualization
    reduced_feats = reduce_dim(matrix=embed, dim=2, method='umap')

    # Use DBSCAN for clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(embed if type_ == 'raw' else reduced_feats)
    cluster_labels = dbscan.labels_

    # Check clustering results
    print("Unique cluster labels:", np.unique(cluster_labels))

    # Count the occurrences of each transcription in each cluster
    class_names = {}
    for i, label in enumerate(pred_seq):
        label = str(label.tolist()) if isinstance(label, torch.Tensor) else label
        if cluster_labels[i] == -1:
            continue  # Skip noise points
        if cluster_labels[i] not in class_names:
            class_names[cluster_labels[i]] = [label]
        else:
            class_names[cluster_labels[i]].append(label)

    unique_class_names = {}
    while class_names:
        most_common_label, most_common_count, most_common_cluster = None, 0, None
        for cluster_label, labels in class_names.items():
            label_counts = {label: labels.count(label) for label in labels}
            common_label, common_count = max(label_counts.items(), key=lambda item: item[1])
            if common_count > most_common_count:
                most_common_label, most_common_count, most_common_cluster = common_label, common_count, cluster_label

        unique_class_names[most_common_cluster] = most_common_label

        # Remove the most common label from all clusters
        for cluster_label in list(class_names.keys()):
            if cluster_label in class_names:
                class_names[cluster_label] = [label for label in class_names[cluster_label] if label != most_common_label]
                if not class_names[cluster_label]:
                    del class_names[cluster_label]

    # Visualize clustering results
    plt.figure(figsize=(8, 8))  # Make the figure square
    for cluster_label, class_name in unique_class_names.items():
        indices = np.where(cluster_labels == cluster_label)
        plt.scatter(reduced_feats[indices, 0], reduced_feats[indices, 1], label=class_name, alpha=0.7, s=100)  # Increase marker size

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 20})  # Move legend outside the plot
    plt.title('Tone Dialect Clustering with Auto-Classification', fontsize=14)  # Increase title font size
    plt.show()

    return unique_class_names