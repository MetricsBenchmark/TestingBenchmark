import cluster
from knee_point import select_optimal_k
from cuml import PCA
import cuml
from cuml import AgglomerativeClustering
from cuml import DBSCAN
from draw_heatmap import heatmap_sorted_by_cluster_similarity

def heatmap_cluster(FE, CA, DR, model_name, fullX, test_set='mnist'):
    import cuml
    import cluster
    from draw_heatmap import heatmap_sorted_by_cluster_similarity
    from DBCV.DBCV import DBCV
    from cuml import DBSCAN
    from torchvision import models
    from cuml import AgglomerativeClustering
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    if 'mnist' in test_set:
        mispred_indices = np.load(f'./result/mnist/mispred_indices/{model_name}_misprd_indices.npy')
    elif 'udacity' in test_set:
        mispred_indices = np.load(f'./result/udacity/0.1308996938995747/{model}_mispred_indices.npy')
    images = fullX[mispred_indices]
    if FE=='resnet50':
        feature = cluster.extract_features_torch(images=images, dataset=test_set)
    else:
        raise NotImplementedError("other FEs are not implemented yet.")
    suite_feat = np.vstack(feature)

    if DR=='UMAP':
        u = cluster.umap_gpu(ip_mat=suite_feat, min_dist=0.1, n_components=min(25, len(suite_feat) - 1), n_neighbors=15, metric='Euclidean')
    elif DR=='PCA':
        from cuml import PCA
        pca_float = PCA(n_components=25)
        u = pca_float.fit_transform(suite_feat)
    elif DR == 'GRP':
        from cuml.random_projection import GaussianRandomProjection
        GRP_float = GaussianRandomProjection(n_components=25, random_state=42)
        u = GRP_float.fit_transform(suite_feat)
    else:
        raise NotImplementedError("other DRs are not implemented yet.")

    if CA=='dbscan':
        optimal_eps = cluster.find_optimal_eps(u)
        dbscan_float = DBSCAN(eps=optimal_eps,
                              # we use different optimal eps across settings due to large variation.
                              min_samples=2)
        labels = dbscan_float.fit_predict(u)
    elif CA == 'hdbscan':
        hdbscan_float = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = hdbscan_float.fit_predict(u)
    elif CA=='HAC':
        HAC = AgglomerativeClustering(n_clusters=10)
        labels = HAC.fit_predict(u)
    elif CA=='Kmeans':
        from cuml.cluster import KMeans
        kmeans_float = KMeans(n_clusters=10)
        labels = kmeans_float.fit_predict(u)

    heatmap_sorted_by_cluster_similarity(u, labels,
    f'./heatmap/save/heatmap_{FE}_{DR}_{CA}.png')
    # np.save(f'./result/cluster_lbls/{model_name}_cluster_lbls.npy', labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST experiments")
    parser.add_argument(
        "--FE",
        type=str,
        default=None,
        help="Feature Extractor"
    )
    parser.add_argument(
        "--DR",
        type=str,
        default=None,
        help="Dimensionality Reduction Algorithm"
    )
    parser.add_argument(
        "--CA",
        type=str,
        default=None,
        help="Clustering Algorithm"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (only needed for retrain_cluster)"
    )
    heatmap_cluster(args.FE, args.CA, args.DR, args.model, testX) # original testing set of MNIST or Udacity.