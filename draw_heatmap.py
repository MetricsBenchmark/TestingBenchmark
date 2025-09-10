import numpy as np
import os
import seaborn as sns
plot_kwds = {'alpha': 0.15, 's': 80, 'linewidths': 0}
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
def heatmap_sorted_by_cluster_similarity(feature, cluster_id, save_path):
    cluster_id = np.array(cluster_id)
    feature = np.array(feature)

    # Separate -1 (noise) label
    non_outlier_mask = cluster_id != -1
    outlier_mask = cluster_id == -1
    non_outlier_labels = np.unique(cluster_id[non_outlier_mask])

    # Reorder clusters by similarity using centroids
    centroids = np.array([feature[cluster_id == lbl].mean(axis=0) for lbl in non_outlier_labels])
    dist = pairwise_distances(centroids)
    ordered_indices = leaves_list(linkage(dist))
    reordered_labels = [non_outlier_labels[i] for i in ordered_indices]

    # Assign new cluster IDs: 0, 1, ..., N (except -1)
    new_label_map = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(reordered_labels)}
    new_label_map[-1] = -1  # keep -1 unchanged
    # Apply new cluster IDs
    updated_labels = np.array([new_label_map[lbl] for lbl in cluster_id])
    sorted_indices = np.argsort(updated_labels)
    u_sorted = feature[sorted_indices][:int(len(feature)*0.5)]
    updated_labels_sorted = updated_labels[sorted_indices][:int(len(updated_labels)*0.5)]

    # Color map for label strip
    unique_display_labels = sorted(set(updated_labels_sorted))
    n_colors = len(unique_display_labels)
    cmap = ListedColormap(plt.cm.get_cmap('tab20', n_colors)(np.linspace(0, 1, n_colors)))
    label_to_color_idx = {lbl: idx for idx, lbl in enumerate(unique_display_labels)}
    color_idx_array = np.vectorize(label_to_color_idx.get)(updated_labels_sorted).reshape(-1, 1)

    # Plot
    plt.rcParams["font.family"] = "STIXGeneral"
    font_size = 30
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)

    # Main heatmap
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(u_sorted, cmap='Spectral', cbar=False, ax=ax0, yticklabels=False)
    ax0.set_title('Heatmap of Features Sorted by Cluster Similarity', fontsize=font_size)
    ax0.set_xlabel('Feature Index', fontsize=font_size)
    x_min, x_max = ax0.get_xlim()
    tick_positions = np.arange(0, x_max + 1, 10)
    ax0.set_xticks(tick_positions)
    ax0.set_xticklabels([int(tick) for tick in tick_positions], fontsize=font_size - 5)
    y_min, y_max = ax0.get_ylim()
    tick_positions = np.arange(0, y_max + 1, 5)
    ax0.set_yticks(tick_positions)
    ax0.set_yticklabels([int(tick) for tick in tick_positions], fontsize=font_size - 5)
    ax0.set_ylabel('Samples', fontsize=font_size)

    # Sidebar showing cluster labels
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(color_idx_array, aspect='auto', cmap=cmap)
    ax1.set_xticklabels([])
    ax1.set_yticks([])

    # Legend with new cluster IDs
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(n_colors)]
    legend_labels = [f'{lbl}' for lbl in unique_display_labels]
    ax1.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='medium', title='Cluster ID')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # visualize single cluster's heatmap only
    create_individual_cluster_heatmaps(feature, updated_labels, save_path)

def create_individual_cluster_heatmaps(feature, cluster_labels, main_save_path):
    """
    Create individual heatmaps for each cluster
    """
    # Get unique cluster labels (excluding noise label -1)
    unique_clusters = sorted(set(cluster_labels))
    unique_clusters = [clust for clust in unique_clusters if clust != -1]

    # Create directory for individual cluster heatmaps
    base_dir = os.path.dirname(main_save_path)
    base_name = os.path.basename(main_save_path).split('.')[0]
    cluster_dir = os.path.join(base_dir, f"{base_name}_individual_clusters")
    os.makedirs(cluster_dir, exist_ok=True)

    plt.rcParams["font.family"] = "STIXGeneral"
    font_size = 30

    for cluster_id in unique_clusters:
        # Get samples belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_features = feature[cluster_mask]

        if len(cluster_features) == 0:
            continue

        # Sort samples within cluster by their feature values (optional)
        # You can remove this sorting if you want to preserve the original order
        sorted_indices = np.argsort(cluster_features.sum(axis=1))
        cluster_features_sorted = cluster_features[sorted_indices]

        # Create heatmap for this cluster
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(cluster_features_sorted,
                    cmap='Spectral',
                    cbar=True,
                    ax=ax,
                    yticklabels=False,
                    xticklabels=False if cluster_features_sorted.shape[1] > 50 else True)

        # Customize plot
        # ax.set_title(f'Cluster {cluster_id} Heatmap\n({len(cluster_features_sorted)} samples)',
        #              fontsize=font_size, pad=20)
        ax.set_xlabel('Feature Index', fontsize=font_size)
        x_min, x_max = ax.get_xlim()
        tick_positions = np.arange(0, x_max + 1, 10)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([int(tick) for tick in tick_positions], fontsize=font_size - 5)
        ax.set_ylabel('Samples', fontsize=font_size)
        # Get the number of samples (rows)
        n_samples = cluster_features_sorted.shape[0]
        # Choose a step size (e.g., every 10th sample)
        step = 5
        # Create tick positions and labels only for every 'step' samples
        tick_positions = np.arange(0, n_samples, step) + 0.5  # Position at cell center
        tick_labels = np.arange(0, n_samples, step)  # Label with the index
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=font_size - 5)

        # Add colorbar label
        cbar = ax.collections[0].colorbar
        cbar.set_label('Feature Value', fontsize=font_size)
        # Set tick font size for the colorbar
        cbar.ax.tick_params(labelsize=font_size-5)

        plt.tight_layout()

        # Save individual cluster heatmap
        cluster_save_path = os.path.join(cluster_dir, f"cluster_{cluster_id}_heatmap.png")
        plt.savefig(cluster_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved heatmap for cluster {cluster_id} with {len(cluster_features)} samples")
