import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from collections import defaultdict


def plot_umap_embeddings(shape_file, label_file, output_file=None, highlight_indices=None):
    shape_array = np.load(shape_file)
    identity_ids = np.load(label_file)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(identity_ids)
    label_names = label_encoder.classes_

    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(shape_array)

    plt.figure(figsize=(20, 20))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=labels, cmap='tab20', s=30, alpha=0.8
    )

    # Highlight misclassified points if provided
    if highlight_indices is not None:
        highlight = embeddings_2d[highlight_indices]
        plt.scatter(highlight[:, 0], highlight[:, 1],
                    edgecolors='red', facecolors='none', s=100, linewidths=1.5, label="Misclassified")

    plt.title("UMAP of FLAME Shape Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    handles = [plt.Line2D([0], [0], marker='o', color='w',
               label=label_names[i],
               markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=6)
               for i in range(len(label_names))]
    if highlight_indices is not None:
        handles.append(plt.Line2D([0], [0], marker='o', color='r', label='Misclassified',
                                  markerfacecolor='none', markersize=8, linewidth=1.5))
    plt.legend(handles=handles, title="Identities", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()



def compute_rr1(shape_array, identity_ids):
    """
    Computes Rank-1 and Rank-5 Recognition Accuracy and tracks misclassified indices.
    """
    grouped = defaultdict(list)
    for shape, identity in zip(shape_array, identity_ids):
        grouped[identity].append(shape)

    gallery = []
    gallery_labels = []
    queries = []
    query_labels = []
    query_indices = []

    index = 0
    for identity, shapes in grouped.items():
        if len(shapes) < 2:
            continue

        shapes = np.array(shapes)
        gallery.append(shapes[0])
        gallery_labels.append(identity)

        for i in range(1, len(shapes)):
            queries.append(shapes[i])
            query_labels.append(identity)
            query_indices.append(index + i)  # index in original shape_array

        index += len(shapes)

    gallery = np.stack(gallery)
    queries = np.stack(queries)
    gallery_labels = np.array(gallery_labels)
    query_labels = np.array(query_labels)

    dists = cosine_distances(queries, gallery)
    nearest_indices = np.argmin(dists, axis=1)
    predicted_labels = gallery_labels[nearest_indices]

    correct = predicted_labels == query_labels
    rr1 = np.mean(correct)
    top_k = 5
    top_k_indices = np.argsort(dists, axis=1)[:, :top_k]
    correct_top_k = np.array([query_labels[i] in gallery_labels[top_k_indices[i]] for i in range(len(query_labels))])
    top_k_acc = np.mean(correct_top_k)

    # Return indices in original array that were misclassified
    misclassified_indices = [query_indices[i] for i, is_correct in enumerate(correct) if not is_correct]

    return rr1, top_k_acc, misclassified_indices


if __name__ == "__main__":
    shapes = np.load("shape_array.npy")
    shapes = normalize(shapes, norm='l2')
    ids = np.load("identity_ids.npy")

    rr1, top_k_acc, misclassified = compute_rr1(shapes, ids)
    print(f"Recognition Accuracy: RR1: {rr1:.4f} , RR5: {top_k_acc:.4f}")
    print(f"Misclassified indices: {misclassified}")

    plot_umap_embeddings("shape_array.npy", "identity_ids.npy", "embeddingy.jpg", highlight_indices=misclassified)

