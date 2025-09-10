
import driving.selection as selection

import sys
import metrics
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from driving.data_utils import load_train_data
import time

basedir = os.path.abspath(os.path.dirname(__file__)) + '/driving/'

out_csv = 'report/driving.csv'
test_dir = 'test/driving'
budgets = [50, 100, 150, 200]

def get_driving_models():
    import driving.driving_models as driving_models
    import driving.epoch.epoch_model as epoch_model
    models = {
        'dave2v1': driving_models.Dave_orig(load_weights=True),
        'dave2v2': driving_models.Dave_norminit(load_weights=True),
        'dave2v3': driving_models.Dave_dropout(load_weights=True),
        'epoch': epoch_model.build_cnn(weights_path='./epoch.h5')
    }
    return models

def from_generator(gen, tot):
    xs, ys = [], []
    collected = 0
    while collected < tot:
        x_batch, y_batch = next(gen)
        remain = tot - collected
        if len(x_batch) > remain:
            x_batch = x_batch[:remain]
            y_batch = y_batch[:remain]
            collected += remain
        else:
            collected += len(x_batch)
        xs.append(x_batch)
        ys.append(y_batch)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def get_datasets(vs):
    datasets = dict()
    for v in vs:
        gen, tot = selection.get_data(v)
        x, y = from_generator(gen, tot)
        datasets[v] = (x, y, tot)
        print('loaded dataset', v)
    return datasets


def get_datapaths(vs):
    xs = []
    ys = []
    if vs == 'udacity':
        with open(basedir + '/testing' + '/CH2_final_evaluation.csv', 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.split(',')
                xs.append(basedir + '/testing' + '/center/' + parts[0] + '.jpg')
                ys.append(float(parts[1]))
        xs, ys = np.array(xs), np.array(ys)
    elif vs == 'udacity_C':
        with open(basedir + '/data' + '/Udacity_C_clean_labeled.txt', 'r') as f:
            for i, line in enumerate(f):
                xs.append(line.split(',')[0])
                ys.append(float(line.split(',')[1]))
        xs, ys = np.array(xs), np.array(ys)
    elif vs == 'udacity_label':
        with open(basedir + '/data' + '/udacity_label_shifted.txt', 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                xs.append(basedir + '/testing' + '/center/' + line.split(',')[0] + '.jpg')
                ys.append(float(line.split(',')[1]))
        xs, ys = np.array(xs), np.array(ys)
    elif vs == 'udacity_dave':
        with open(basedir + '/data' + '/udacity_dave.txt', 'r') as f:
            for i, line in enumerate(f):
                xs.append(line.split(',')[0])
                ys.append(float(line.split(',')[1]))
        xs, ys = np.array(xs), np.array(ys)
    elif vs == 'udacity_adv':
        xs = np.load(basedir + '/data/fgsm_bim_pgd_clean_udacity_eps8_image.npy')
        ys = np.load(basedir + '/data/fgsm_bim_pgd_clean_udacity_eps8_label.npy')

    return xs, ys

def run_selection(model_name, test_set, datasets, metricList, budgets):
    models = get_driving_models()
    train_gen, tot = selection.get_train_data()
    trainX, trainy = from_generator(train_gen, tot)
    model = models[model_name]
    testX, testy, tot = datasets[test_set]

    for m in metricList:
        for b in budgets:
            selectedX, selectedy, idx = metrics.select(trainX, trainy, testX, testy, model, b, m, test_set, model_name)
            test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
            if not os.path.exists(test_out_dir):
                os.makedirs(test_out_dir)
            np.savetxt(os.path.join(test_out_dir, 'X.txt'), idx, fmt='%d')
            np.savetxt(os.path.join(test_out_dir, 'y.txt'), selectedy, fmt='%f')

def select():
    models = get_driving_models()
    metricList = ['dr', 'ces', 'rnd', 'gd', 'kmnc', 'nac', 'std', 'pace', 'est', 'lsa']
    datasets = get_datasets(['udacity'])
    for m in models.keys():
        run_selection(m, 'udacity', datasets, metricList, budgets)

    for v in ['udacity_C', 'udacity_label', 'udacity_adv', 'udacity_dave']:
        datasets = get_datasets([v])
        run_selection('epoch', v, datasets, metricList, budgets)

def time_cost(model_name, test_set, datasets, metricList):
    train_gen, tot = selection.get_train_data()
    trainX, trainy = from_generator(train_gen, tot)
    b = 200
    models = get_driving_models()
    model = models[model_name]
    testX, testy, tot = datasets[test_set]
    for m in metricList:
        start = time.time()
        selectedX, selectedy, idx = metrics.select(trainX, trainy,
                                                   testX, testy, model, b, m, test_set, model_name
                                                   )
        time_cost = time.time() - start
        print('time cost', time_cost)

def time_efficiency():
    metricList = ['dr', 'ces', 'rnd', 'gd', 'kmnc', 'nac', 'std', 'pace', 'est', 'lsa']
    models = get_driving_models()
    datasets = get_datasets(['udacity'])
    for m in models.keys():
        time_cost(m, 'udacity', datasets, metricList)

    for v in ['udacity_C', 'udacity_label', 'udacity_adv', 'udacity_dave']:
        datasets = get_datasets([v])
        time_cost('epoch', v, datasets, metricList)


def evaluate_cluster():
    vals = []
    metricList = ['dr', 'ces', 'est', 'rnd', 'gd', 'kmnc', 'nac', 'lsa', 'std', 'pace']

    xs, _ = get_datapaths('udacity')
    for m in ['dave2v1', 'dave2v2', 'dave2v3', 'epoch']:
        vals.extend(evaluation_cluster(m, 'udacity', metricList, budgets, xs))

    for v in ['udacity_label', 'udacity_adv', 'udacity_dave']:  # 'udacity_C',
        xs, _ = get_datapaths(v)
        vals.extend(evaluation_cluster('epoch', v, metricList, budgets, xs))

def evaluation_cluster(model_name, test_set, metricList, budgets, fullX):
    from DBCV.DBCV import DBCV
    import cuml
    from cuml import DBSCAN
    from torchvision import models
    import cluster
    import sklearn
    results = []
    thresholds = np.arange(0, 25.1, 2.5) / 180 * np.math.pi
    for threshold in thresholds[:-3]:
        for m in metricList:
            for b in budgets:
                test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
                if not os.path.exists(test_out_dir):
                    raise FileNotFoundError(f"Test output directory {test_out_dir} does not exist.")
                mispred_indices = np.load(f'./test/driving/cluster_idx/{threshold}/{test_set}_{model_name}_{m}_{b}.npy')
                if len(mispred_indices) <= 2:
                    clustering_results = {"Number of Clusters": -1,
                                          "Silhouette Score": -1.0,
                                          "DBCV Score": -1,
                                          "Combined Score": -1.0,
                                          "Number of Mispredicted Inputs": len(mispred_indices),
                                          "Number of Noisy Inputs": -1,
                                          "test_set": test_set,
                                          "selection_metric": m,
                                          "budget": b,
                                          "model": model_name,
                                          "threshold": np.round(threshold, 3)}
                else:

                    paths = fullX[mispred_indices]
                    feature = cluster.extract_features_torch(images=paths, dataset=test_set)
                    suite_feat = np.vstack(feature)

                    u = cluster.umap_gpu(ip_mat=suite_feat, min_dist=0.1, n_components=min(25, len(suite_feat) - 1), n_neighbors=15,
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
                            "model": model_name,
                            "threshold": np.round(threshold, 3)
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
                            "model": model_name,
                            "threshold": np.round(threshold, 3)
                        }
                print(clustering_results)
                results.append(clustering_results)
    return results

def run_evaluation(model_name, test_set, metricList, budgets, fullX, fully, originalX, originaly):
    models = get_driving_models()
    model = models[model_name]
    full_pred = model.predict(fullX, verbose=0).flatten()

    results = []

    for m in metricList:
        for b in budgets:
            test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
            if not os.path.exists(test_out_dir):
                raise FileNotFoundError(f"Test output directory {test_out_dir} does not exist.")
            X_id = np.loadtxt(os.path.join(test_out_dir, 'X.txt'), dtype=int)
            sort = np.zeros(fullX.shape[0], dtype=int)
            sort[X_id] = np.arange(1, b + 1)

            thresholds = np.arange(0, 25.1, 2.5) / 180 * np.math.pi # transform degree intervals to radians
            failures = [np.sum(np.abs(full_pred[X_id] - fully[X_id]) > thresholds[i]) for i in range(len(thresholds))]

            acc_hat = 1-np.sqrt(np.mean(np.square(full_pred[X_id] - fully[X_id])))
            acc = 1-np.sqrt(np.mean(np.square(full_pred - fully)))
            rmse_score = np.abs(acc_hat - acc)
            # concatenatedX, concatenatedy = fullX[X_id], fully[X_id]
            # Type 2 retraining
            concatenatedX = np.concatenate((originalX, fullX[X_id]), axis=0)
            concatenatedy = np.concatenate((originaly, fully[X_id]), axis=0)
            model.compile(loss="mse", optimizer="adadelta")
            model.fit(concatenatedX, concatenatedy, epochs=5, batch_size=128, verbose=0)

            mask = np.ones(fullX.shape[0], dtype=bool)
            mask[X_id] = False
            retrain_pred = model.predict(fullX[mask], verbose=0).flatten()

            retrain_acc = 1-np.sqrt(np.mean(np.square(retrain_pred - fully[mask])))
            acc_clean = 1-np.sqrt(np.mean(np.square(full_pred[mask] - fully[mask])))
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
    models = get_driving_models()
    vals = []
    train_gen, tot = selection.get_train_data()
    trainX, trainy = from_generator(train_gen, tot) # preprocessed train data.
    originalX, originaly = trainX, trainy
    metricList = ['dr', 'ces', 'est', 'rnd', 'ent', 'gini', 'gd', 'kmnc', 'nac', 'lsa', 'std', 'pace']

    datasets = get_datasets(['udacity'])
    testX, testy, _ = datasets['udacity']
    for m in models.keys():
        vals.extend(run_evaluation(m, 'udacity', metricList, budgets, testX, testy, originalX, originaly))

    datasets = get_datasets(['udacity_C'])
    _X, _y, _ = datasets['udacity_C']
    vals.extend(run_evaluation('epoch', 'udacity_C', metricList, budgets, _X, _y, originalX, originaly))

    datasets = get_datasets(['udacity_adv'])
    _X, _y, _ = datasets['udacity_adv']
    vals.extend(run_evaluation('epoch', 'udacity_adv', metricList, budgets, _X, _y, originalX, originaly))

    datasets = get_datasets(['udacity_label'])
    _X, _y, _ = datasets['udacity_label']
    vals.extend(run_evaluation('epoch', 'udacity_label', metricList, budgets, _X, _y, originalX, originaly))

    datasets = get_datasets(['udacity_dave'])
    _X, _y, _ = datasets['udacity_dave']
    vals.extend(run_evaluation('epoch', 'udacity_dave', metricList, budgets, _X, _y, originalX, originaly))

def run_save():
    models = get_driving_models()
    datasets = get_datasets(['udacity'])
    testX, testy, _ = datasets['udacity']
    for model_name in models.keys():
        save_idx(model_name, testX, testy, 'udacity')
    for v in ['udacity_C', 'udacity_label', 'udacity_adv', 'udacity_dave']:
        datasets = get_datasets([v])
        testX, testy, tot = datasets[v]
        save_idx('epoch', testX, testy, v)

def save_idx(model_name, fullX, fully, test_set):
    metricList = ['dr', 'ces', 'est', 'rnd', 'ent', 'gini', 'gd', 'kmnc', 'nac', 'lsa', 'std', 'pace']
    models = get_driving_models()
    model = models[model_name]
    full_pred = model.predict(fullX, verbose=0).flatten()
    for m in metricList:
        for b in budgets:
            test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
            X_id = np.loadtxt(os.path.join(test_out_dir, 'X.txt'), dtype=int)  # (n_selected,)
            thresholds = np.arange(0, 25.1, 2.5) / 180 * np.math.pi  # transform degree intervals to radians
            for i in range(len(thresholds)):
                idx = np.abs(full_pred[X_id] - fully[X_id]) > thresholds[i]
                mispred_indices = X_id[idx]
                root = f'./test/driving/cluster_idx/{thresholds[i]}/'
                if not os.path.exists(root):
                    os.makedirs(root)
                np.save(root+f'{test_set}_{model_name}_{m}_{b}.npy', mispred_indices)

def rmse(acc, acc_hat, randomness=True):
    if randomness:
        N = len(acc)
        return np.sqrt(1 / N * np.sum((acc_hat - acc) ** 2, axis=1))
    else:
        return np.abs(acc_hat - acc)


def save_mispred_indices(): # save mispred indices for testX and testy
    datasets = get_datasets(['udacity'])
    testX, testy, tot = datasets['udacity'] # preprocessed (5614,100,100,3); (5614,) float
    models = get_driving_models()
    for model_name in models.keys():
        models = get_driving_models()
        model = models[model_name]
        full_pred = model.predict(testX, verbose=0).flatten()
        thresholds = np.arange(0, 25.1, 2.5) / 180 * np.math.pi  # transform degree intervals to radians
        for i in range(len(thresholds)):
            idx = np.abs(full_pred - testy) > thresholds[i]
            mispred_indices = idx
            root = f'./result/udacity/{thresholds[i]}/'
            if not os.path.exists(root):
                os.makedirs(root)
            np.save(root + f'{model_name}_mispred_indices.npy', mispred_indices)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Udacity experiments")
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
