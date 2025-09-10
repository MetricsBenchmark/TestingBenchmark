
import mnist_cifar_imagenet_svhn.selection as mnist
import sys
import metrics
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import sklearn
import time

model_names = ['lenet1', 'lenet4', 'lenet5']
testX, testy = mnist.get_data('lenet1')
trainX, trainy = mnist.get_mnist_train()


out_csv = 'report/mnist.csv'
test_dir = 'test/mnist'

budgets = [50, 100, 150, 200]

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

def run_selection(model_name, test_set, testX, testy, metricList, budgets):
    model = mnist.get_model(model_name)
    for m in metricList:
        for b in budgets:
            test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
            if m == 'dat':
                n_test = testX.shape[0]
                hybridX = np.concatenate((trainX[:int(n_test // 2)], testX[:int(n_test // 2)]), axis=0)
                hybridy = np.concatenate((trainy[:int(n_test // 2)], testy[:int(n_test // 2)]), axis=0)
                selectedX, selectedy, idx = metrics.dat_ood_detector(
                    testX, testy, model, b, test_set, trainX, trainy, hybridX, hybridy, batch_size=128, num_classes=10
                )
            else:
                selectedX, selectedy, idx = metrics.select(trainX, trainy,
                    testX, testy, model, b, m, test_set, model_name
                )
            np.savetxt(os.path.join(test_out_dir, 'X.txt'), idx, fmt='%d')
            np.savetxt(os.path.join(test_out_dir, 'y.txt'), onehot_to_int(selectedy).astype(int), fmt='%d')


def select():
    metricList = ['dr', 'ces', 'mcp', 'est', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'std', 'pace']

    for m in model_names:
        run_selection(m, 'mnist', testX, testy, metricList, budgets)

    _X, _y = mnist.get_corrupted_mnist()
    run_selection('lenet5', 'mnist_c', _X, _y, metricList, budgets)

    _X, _y = mnist.get_adv_mnist()
    run_selection('lenet5', 'mnist_adv', _X, _y, metricList, budgets)

    _X, _y = mnist.get_label_mnist()
    run_selection('lenet5', 'mnist_label', _X, _y, metricList, budgets)

    _X, _y = mnist.get_mnist_emnist()
    run_selection('lenet5', 'mnist_emnist', _X, _y, metricList, budgets)

def time_cost(model_name, test_set, testX, testy, metricList, csv='report/mnist_time.csv'):
    b = 200
    model = mnist.get_model(model_name)
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
        print('time cost', time_cost)


def time_efficiency():
    metricList = ['dr', 'ces', 'mcp', 'est', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'std', 'pace']

    for m in model_names:
        time_cost(m, 'mnist', testX, testy, metricList)

    _X, _y = mnist.get_corrupted_mnist()
    time_cost('lenet5', 'mnist_c', _X, _y, metricList)

    _X, _y = mnist.get_adv_mnist()
    time_cost('lenet5', 'mnist_adv', _X, _y, metricList)

    _X, _y = mnist.get_label_mnist()
    time_cost('lenet5', 'mnist_label', _X, _y, metricList)

    _X, _y = mnist.get_mnist_emnist()
    time_cost('lenet5', 'mnist_emnist', _X, _y, metricList)


def rmse(acc, acc_hat, randomness=True):
    if randomness:
        N = len(acc)
        return np.sqrt(1 / N * np.sum((acc_hat - acc) ** 2, axis=1))
    else:
        return np.abs(acc_hat - acc)

def run_evaluation(model_name, test_set, metricList, budgets, fullX, fully, originalX, originaly):
    model = mnist.get_model(model_name)
    full_y_int = onehot_to_int(fully)
    full_pred = model.predict(fullX, verbose=0)
    full_pred_int = np.argmax(full_pred, axis=1)

    results = []
    for m in metricList:
        for b in budgets:
            test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
            X_id = np.loadtxt(os.path.join(test_out_dir, 'X.txt'), dtype=int)
            sort = np.zeros(fullX.shape[0], dtype=int)
            sort[X_id] = np.arange(1, b + 1)

            acc_hat = np.mean(full_pred_int[X_id] == full_y_int[X_id])
            failures = np.sum(full_pred_int[X_id] != full_y_int[X_id])

            acc = np.mean(full_pred_int == full_y_int)
            rmse_score = np.abs(acc_hat - acc)

            # Type 2 retraining
            concatenatedX = np.concatenate((originalX, fullX[X_id]), axis=0)
            concatenatedy = np.concatenate((originaly, fully[X_id]), axis=0)
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
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
        vals.extend(run_evaluation(m, 'mnist', metricList, budgets, testX, testy, originalX, originaly))

    _X, _y = mnist.get_corrupted_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_c', metricList, budgets, _X, _y, originalX, originaly))

    _X, _y = mnist.get_adv_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_adv', metricList, budgets, _X, _y, originalX, originaly))

    _X, _y = mnist.get_label_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_label', metricList, budgets, _X, _y, originalX, originaly))

    _X, _y = mnist.get_mnist_emnist()
    vals.extend(run_evaluation('lenet5', 'mnist_emnist', metricList, budgets, _X, _y, originalX, originaly))

def evaluate_cluster():
    vals = []
    metricList = ['dr', 'ces', 'mcp', 'est', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'std', 'pace']

    for m in model_names:
        vals.extend(evaluation_cluster(m, 'mnist', metricList, budgets, testX, testy))

    _X, _y = mnist.get_corrupted_mnist()
    vals.extend(evaluation_cluster('lenet5', 'mnist_c', metricList, budgets, _X, _y))

    _X, _y = mnist.get_adv_mnist()
    vals.extend(evaluation_cluster('lenet5', 'mnist_adv', metricList, budgets, _X, _y))

    _X, _y = mnist.get_label_mnist()
    vals.extend(evaluation_cluster('lenet5', 'mnist_label', metricList, budgets, _X, _y))

    _X, _y = mnist.get_mnist_emnist()
    vals.extend(evaluation_cluster('lenet5', 'mnist_emnist', metricList, budgets, _X, _y))

def evaluation_cluster(model_name, test_set, metricList, budgets, fullX, fully):
    from cuml.manifold import UMAP
    from DBCV.DBCV import DBCV
    import cuml
    from cuml import DBSCAN
    from torchvision import models
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    model_resnet50 = FeatureExtractor(resnet50)
    results = []
    for m in metricList:
        for b in budgets:
            mispred_indices = np.load(f'./test/mnist/cluster_idx/{test_set}_{model_name}_{m}_{b}.npy')
            if len(mispred_indices) <= 2:
                # print(f'Insufficient mispredictions for idr={idr}, dataset={dataset}, model={model_name}, bg={bg}')
                clustering_results = {"Number of Clusters": -1,
                                      "Silhouette Score": -1.0,
                                      "Combined Score": -1.0,
                                      "Number of Mispredicted Inputs": len(mispred_indices),
                                      "Number of Noisy Inputs": -1,
                                      "Config": -1,
                                      "test_set": test_set,
                                      "selection_metric": m,
                                      "budget": b,
                                      "model": model_name}
            else:
                images = fullX[mispred_indices]
                feature = extract_features_torch(images=images, model=model_resnet50, dataset_name=test_set)
                suite_feat = np.vstack(feature)

                u = umap_gpu(ip_mat=suite_feat, min_dist=0.1, n_components=min(25, len(suite_feat) - 1), n_neighbors=15,
                             metric='Euclidean')
                optimal_eps = find_optimal_eps(u)
                dbscan_float = DBSCAN(eps=optimal_eps, min_samples=2)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST experiments")
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
