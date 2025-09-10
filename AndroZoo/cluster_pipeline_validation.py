def feature_extract(FE, model, dataset):
    # use deepdrebin trained on Drebin dataset for feature extraction
    path = f'./{dataset}_test_suite/'
    X_feat = readdata_np(path + f'{FE}/{dataset}_test_dense1.data')
    y_pred = readdata_np(path + f'{model}/{dataset}_test_pred.data')
    testy = readdata_np(path + f'{model}/{dataset}_test_y.data')
    mis_idx = testy != y_pred
    X = X_feat[mis_idx]
    gt_lbls = testy[mis_idx]
    pred_lbls = y_pred[mis_idx]
    suite_feat = np.vstack(X)
    return suite_feat, (X, mis_idx, gt_lbls, pred_lbls)

def dimension_reduce(DR, suite_feat):
    if DR == 'UMAP':
        u = umap_gpu(ip_mat=suite_feat, min_dist=0.1, n_components=25, n_neighbors=15,
                     metric='Euclidean')
    elif DR == 'PCA':
        pca_float = PCA(n_components=25)
        u = pca_float.fit_transform(suite_feat)
    elif DR == 'GRP':
        GRP_float = GaussianRandomProjection(n_components=25, random_state=42)
        u = GRP_float.fit_transform(suite_feat)
    return u
def clustering(Clustering, u):
    if Clustering == 'HDBSCAN':
        # for min_cluster_size, we set to 2 due to the small numer of mispredicted pts.
        hdbscan_float = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = hdbscan_float.fit_predict(u)
    elif Clustering == 'DBSCAN':
        dbscan_float = DBSCAN(eps=0.603, min_samples=2)
        labels = dbscan_float.fit_predict(u)
    elif Clustering == 'HAC':
        HAC = AgglomerativeClustering(n_clusters=10)
        labels = HAC.fit_predict(u)
    elif Clustering == 'Kmeans':
        kmeans_float = KMeans(n_clusters=9)
        labels = kmeans_float.fit_predict(u)
    return labels

import seaborn as sns
from draw_heatmap import heatmap_sorted_by_cluster_similarity
def test_pipelines(FE, DR, Clustering, dataset, model, save=False):
    # feat_dim must <= num_samples
    suite_feat, (X, index, gt_lbls, pred_lbls) = feature_extract(FE, model, dataset)
    u = dimension_reduce(DR, suite_feat)
    labels = clustering(Clustering, u)
    if save == True:
        save_path = './cluster_data/cluster_save'
        save_path2 = save_path + f'/{model}/{dataset}/{DR}_{Clustering}'
        if not os.path.exists(save_path2):
            os.makedirs(save_path2)
        np.save(save_path2 + '/X_feat_mispredict.npy', X)
        np.save(save_path2 + '/index_mispredict.npy', index)
        np.save(save_path2 + '/gt_lbls_mispredict.npy', gt_lbls)
        np.save(save_path2 + '/pred_lbls_mispredict.npy', pred_lbls)
        np.save(save_path2 + '/reduced_features_u.npy', u)
        np.save(save_path2 + '/cluster_labels.npy', labels)
    heatmap_sorted_by_cluster_similarity(u, labels, f'./heatmap/save/heatmap_similarity_ordered_{DR}_{Clustering}_{model}_{dataset}.png')

    silhouette = sklearn.metrics.silhouette_score(u, labels)
    DBCV_score = DBCV(u, labels)
    print(silhouette, DBCV_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IMDb experiments, test the candidate pipeline")
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
    test_pipelines(args.FE, args.DR, args.CA, 'Andro', args.model)