import sys
import metrics
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
# import cuml
# from cuml import AgglomerativeClustering
# import hdbscan
# from cuml.manifold import UMAP
# from cuml.cluster import KMeans
# import seaborn as sns

metricList = ['rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'nc', 'std', 'pace', 'dr', 'ces', 'mcp', 'est']
budgets = [50, 100, 150, 200]
model_names = ['transformer', 'lstm', 'gru', 'linear']

trainX = np.load('data/tokenized/trainX.npy')
trainy = np.load('data/tokenized/trainy.npy')
testX = np.load('data/tokenized/testX.npy')
testy = np.load('data/tokenized/testy.npy')

test_dir = 'test/imdb'
def onehot_to_int(y):
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    elif y.ndim == 1:
        return y
    else:
        raise ValueError("Input must be a one-hot encoded array or a single-dimensional array.")

def int_to_onehot(y, num_classes=10):
    if y.ndim == 1:
        return np.eye(num_classes)[y]
    elif y.ndim == 2 and y.shape[1] == num_classes:
        return y
    else:
        raise ValueError("Input must be a one-hot encoded array or a single-dimensional array.")

def run_selection(model_name, test_set, testX, testy, metricList, budgets, force_save=False):
    import IMDB_models as imdb
    model = imdb.lm(name=model_name, mode='tokenized')
    for m in metricList:
        for b in budgets:
            test_out_dir = os.path.join('test/imdb', test_set, model_name, m, str(b))

            if m == 'dat':
                n_test = testX.shape[0]
                hybridX = np.concatenate((trainX[:int(n_test // 2)], testX[:int(n_test // 2)]), axis=0)
                hybridy = np.concatenate((trainy[:int(n_test // 2)], testy[:int(n_test // 2)]), axis=0)
                selectedX, selectedy, idx = metrics.dat_ood_detector(
                    testX, testy, model, b, test_set, trainX, trainy, hybridX, hybridy, batch_size=128, num_classes=2
                )
            else:
                selectedX, selectedy, idx = metrics.select(trainX, trainy,
                                                           testX, testy, model, b, m, test_set, model_name
                                                           )

            np.savetxt(os.path.join(test_out_dir, 'X.txt'), idx, fmt='%d')
            np.savetxt(os.path.join(test_out_dir, 'y.txt'), onehot_to_int(selectedy), fmt='%d')

covariateX = np.load('data/covariate/X.npy')
covariatey = np.load('data/covariate/y.npy')

naturalCovariateX = np.load('data/customer/X.npy')
naturalCovariatey = np.load('data/customer/y.npy')

labelX = np.load('data/label/X.npy')
labely = np.load('data/label/y.npy')

advX = np.load('data/adv/X_attacked_clean.npy')
advy = np.load('data/adv/y.npy')

def select():
    metricList = ['dr', 'ces', 'mcp', 'est', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'std', 'pace']

    for m in model_names:
        run_selection(m, 'imdb', testX, testy, metricList, budgets)

    run_selection('transformer', 'imdb_c', covariateX, covariatey, metricList, budgets)
    run_selection('transformer', 'imdb_natural', naturalCovariateX, naturalCovariatey, metricList, budgets)
    run_selection('transformer', 'imdb_label', labelX, labely, metricList, budgets)
    run_selection('transformer', 'imdb_adv', advX, advy, metricList, budgets)

def rmse(acc, acc_hat, randomness=True):
    if randomness:
        N = len(acc)
        return np.sqrt(1 / N * np.sum((acc_hat - acc) ** 2, axis=1))
    else:
        return np.abs(acc_hat - acc)

def run_evaluation(model_name, test_set, metricList, budgets, fullX, fully, originalX, originaly):
    import IMDB_models as imdb
    model = imdb.lm(name=model_name, mode='tokenized')
    full_y_int = onehot_to_int(fully)  # (n_samples,)
    full_pred = model.predict(fullX, verbose=0)  # (n_samples, 2)
    full_pred_int = np.argmax(full_pred, axis=1)  # (n_samples,)

    results = []
    for m in metricList:
        for b in budgets:
            if True:
                test_out_dir = os.path.join('test/imdb', test_set, model_name, m, str(b))
                if not os.path.exists(test_out_dir):
                    raise FileNotFoundError(f"Test output directory {test_out_dir} does not exist.")
                X_id = np.loadtxt(os.path.join(test_out_dir, 'X.txt'), dtype=int)  # (n_selected,)
                sort = np.zeros(fullX.shape[0], dtype=int)
                sort[X_id] = np.arange(1, b + 1)

                acc_hat = np.mean(full_pred_int[X_id] == full_y_int[X_id])
                failures = np.sum(full_pred_int[X_id] != full_y_int[X_id])

                acc = np.mean(full_pred_int == full_y_int)
                rmse_score = np.abs(acc_hat - acc)

                # According to HU etal, we use Type 2 retraining for IMDB
                concatenatedX = np.concatenate((originalX, fullX[X_id]), axis=0)
                concatenatedy = np.concatenate((originaly, fully[X_id]), axis=0)
                model.fit(concatenatedX, concatenatedy, epochs=5, batch_size=128, verbose=0)

                mask = np.ones(fullX.shape[0], dtype=bool)
                mask[X_id] = False
                retrain_pred = model.predict(fullX[mask], verbose=0)
                retrain_pred_int = np.argmax(retrain_pred, axis=1)
                retrain_acc = np.mean(retrain_pred_int == full_y_int[mask])
                acc_clean = np.mean(full_pred_int[mask] == full_y_int[mask])
                acc_improvement = retrain_acc - acc_clean

                re = {
                    'model': model_name,
                    'test_set': test_set,
                    'selection_metric': m,
                    'budget': b,
                    'acc': acc,
                    'acc_hat': acc_hat,
                    'failures': failures,
                    'rmse': rmse_score,
                    'retrain_acc': retrain_acc,
                    'acc_improvement': acc_improvement
                }
                results.append(re)
                print(re)
    return results

def evaluate():
    vals = []
    originalX, originaly = trainX, trainy
    metricList = ['dr', 'ces', 'mcp', 'est', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'std', 'pace']

    for m in model_names:
        vals.extend(run_evaluation(m, 'imdb', metricList, budgets, testX, testy, originalX, originaly))
        save_results(f'{m}_imdb', vals)

    vals.extend(run_evaluation('transformer', 'imdb_c', metricList, budgets, covariateX, covariatey, originalX, originaly))
    vals.extend(run_evaluation('transformer', 'imdb_natural', metricList, budgets, naturalCovariateX, naturalCovariatey, originalX, originaly))
    vals.extend(run_evaluation('transformer', 'imdb_label', metricList, budgets, labelX, labely, originalX, originaly))
    vals.extend(run_evaluation('transformer', 'imdb_adv', metricList, budgets, advX, advy, originalX, originaly))

def evaluation_cluster(model_name, test_set, metricList, budgets):
    from DBCV.DBCV import DBCV
    import cuml
    from cuml import DBSCAN
    from torchvision import models
    import sklearn
    import cluster
    results = []
    for m in metricList:
        for b in budgets:
            test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
            if not os.path.exists(test_out_dir):
                raise FileNotFoundError(f"Test output directory {test_out_dir} does not exist.")
            mispred_indices = np.load(f'./test/imdb/cluster_idx/{test_set}_{model_name}_{m}_{b}.npy')
            if len(mispred_indices) <= 2:
                # print(f'Insufficient mispredictions for idr={idr}, dataset={dataset}, model={model_name}, bg={bg}')
                clustering_results = {"Number of Clusters": 0,
                                      "Silhouette Score": -1.0,
                                      "DBCV Score": -1,
                                      "Combined Score": -1.0,
                                      "Number of Mispredicted Inputs": len(mispred_indices),
                                      "Number of Noisy Inputs": -1,
                                      "test_set": test_set,
                                      "selection_metric": m,
                                      "budget": b,
                                      "model": model_name,
                                      }
            else:
                suite_feat = np.load(f'./test/imdb/cluster_feat/{test_set}_{model_name}_{m}_{b}.npy')
                u = cluster.umap_gpu(ip_mat=suite_feat, min_dist=0.1, n_components=min(50, len(suite_feat) - 1), n_neighbors=15,
                             metric='Euclidean')
                if np.isnan(u).any():
                    breakpoint()
                optimal_eps = cluster.find_optimal_eps(u)
                dbscan_float = DBSCAN(eps=optimal_eps,
                                      # we use different optimal eps across settings due to large variation.
                                      min_samples=2)
                labels = dbscan_float.fit_predict(u)

                if len(np.unique(labels)) == 1:
                    clustering_results = {
                        "Number of Clusters": labels.max() + 1,
                        "Silhouette Score": -1,
                        "DBCV Score": -1,
                        "Combined Score": -1,  # 0.5 * silhouette_umap + 0.5 * DBCV_score,
                        "Number of Mispredicted Inputs": len(u),
                        "Number of Noisy Inputs": list(labels).count(-1),
                        "test_set": test_set,
                        "selection_metric": m,
                        "budget": b,
                        "model": model_name
                    }
                else:
                    silhouette_umap = sklearn.metrics.silhouette_score(u, labels)
                    DBCV_score = DBCV(u, labels)
                    clustering_results = {
                        "Number of Clusters": labels.max() + 1,
                        "Silhouette Score": silhouette_umap,
                        "DBCV Score": DBCV_score,
                        "Combined Score": 0.5 * silhouette_umap + 0.5 * DBCV_score,
                        "Number of Mispredicted Inputs": len(u),
                        "Number of Noisy Inputs": list(labels).count(-1),
                        "test_set": test_set,
                        "selection_metric": m,
                        "budget": b,
                        "model": model_name
                    }
            print(clustering_results)
            results.append(clustering_results)
    return results

def evaluate_cluster():
    vals = []
    metricList = ['dr', 'ces', 'mcp', 'est', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'std', 'pace']
    for m in model_names:
        vals.extend(evaluation_cluster(m, 'imdb', metricList, budgets))

    for v in ['imdb_c', 'imdb_natural', 'imdb_label', 'imdb_adv']:
        vals.extend(evaluation_cluster('transformer', v, metricList, budgets))

def get_ood(name):
    if name == 'imdb_c':
        dataX, datay = covariateX, covariatey
    elif name == 'imdb_natural':
        dataX, datay = naturalCovariateX, naturalCovariatey
    elif name == 'imdb_label':
        dataX, datay = labelX, labely
    elif name == 'imdb_adv':
        dataX, datay = advX, advy
    return dataX, datay

def time_cost(model_name, test_set, testX, testy, metricList, csv='report/imdb_time.csv'):
    import IMDB_models as imdb
    b = 200
    model = imdb.lm(name=model_name, mode='tokenized')
    for m in metricList:
        if m == 'dat':
            n_test = testX.shape[0]
            hybridX = np.concatenate((trainX[:int(n_test // 2)], testX[:int(n_test // 2)]), axis=0)
            hybridy = np.concatenate((trainy[:int(n_test // 2)], testy[:int(n_test // 2)]), axis=0)
            start = time.time()
            selectedX, selectedy, idx = metrics.dat_ood_detector(
                testX, testy, model, b, test_set, trainX, trainy, hybridX, hybridy, batch_size=128,
                num_classes=10
            )
            time_cost = time.time() - start
        else:
            start = time.time()
            selectedX, selectedy, idx = metrics.select(trainX, trainy,
                                                       testX, testy, model, b, m, test_set, model_name
                                                       )
            time_cost = time.time() - start

def time_efficiency():
    metricList = ['dr', 'ces', 'mcp', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'std', 'pace']
    for m in model_names:
        time_cost(m, 'imdb', testX, testy, metricList)
    for v in ['imdb_c', 'imdb_natural', 'imdb_label', 'imdb_adv']:
        _X, _y = get_ood(v)
        time_cost('transformer', v, _X, _y, metricList)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IMDb experiments")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment to run: select, evaluate, evaluate_cluster, time_efficiency"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (only needed for retrain_cluster)"
    )

    args = parser.parse_args()
    if args.exp == "select":
        select()
    elif args.exp == "evaluate":
        evaluate()
    elif args.exp == "evaluate_cluster":
        evaluate_cluster()
    elif args.exp == "time_efficiency":
        time_efficiency()
    else:
        raise ValueError(f"Unknown experiment: {args.exp}")