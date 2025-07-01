import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from collections import defaultdict


def plot_umap_embeddings(shape_file, label_file, output_file=None):
    # Load data
    shape_array = np.load(shape_file)
    identity_ids = np.load(label_file)

    print(f"Loaded {shape_array.shape[0]} embeddings with shape dimension {shape_array.shape[1]}")

    # Encode identity strings to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(identity_ids)
    label_names = label_encoder.classes_

    # Use UMAP to reduce dimensions
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(shape_array)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=labels, cmap='tab20', s=30, alpha=0.8
    )
    plt.title("UMAP of FLAME Shape Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               label=label_names[i],
               markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=6)
               for i in range(len(label_names))]
    plt.legend(handles=handles, title="Identities", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def compute_rr1(shape_array, identity_ids):
    """
    Computes Rank-1 Recognition Accuracy (RR1) using 1:N matching.
    
    Args:
        shape_array: (N, D) numpy array of shape embeddings
        identity_ids: (N,) array of string identity labels

    Returns:
        rr1_accuracy: float
    """
    # Group all embeddings by identity
    grouped = defaultdict(list)
    for shape, identity in zip(shape_array, identity_ids):
        grouped[identity].append(shape)

    gallery = []
    gallery_labels = []
    queries = []
    query_labels = []

    for identity, shapes in grouped.items():
        if len(shapes) < 2:
            continue  # need at least 1 gallery and 1 query

        shapes = np.array(shapes)
        gallery.append(shapes[0])
        gallery_labels.append(identity)

        for i in range(1, len(shapes)):
            queries.append(shapes[i])
            query_labels.append(identity)

    gallery = np.stack(gallery)
    queries = np.stack(queries)
    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)

    print(f"Gallery size: {len(gallery)} | Query size: {len(queries)}")

    # Compute distances (Q x G)
    #dists = euclidean_distances(queries, gallery)
    dists = cosine_distances(queries, gallery)

    # Find closest gallery index per query
    nearest_indices = np.argmin(dists, axis=1)
    predicted_labels = gallery_labels[nearest_indices]

    # Rank-1 Accuracy
    correct = np.sum(predicted_labels == query_labels)
    total = len(query_labels)
    rr1 = correct / total if total > 0 else 0.0

    # Rank-5 Accuracy
    top_k = 5
    top_k_indices = np.argsort(dists, axis=1)[:, :top_k]
    correct_top_k = np.array([query_labels[i] in gallery_labels[top_k_indices[i]] for i in range(len(query_labels))])
    top_k_acc = np.mean(correct_top_k)

    return rr1, top_k_acc

if __name__ == "__main__":
    plot_umap_embeddings("shape_array.npy", "identity_ids.npy", "embeddingy.jpg")

    shapes = np.load("shape_array.npy")
    shapes = normalize(shapes, norm='l2')
    ids = np.load("identity_ids.npy")

    rr1, top_k_acc = compute_rr1(shapes, ids)
    print(f"Recognition Accuracy: RR1: {rr1:.4f} , RR5: {top_k_acc:.4f}")