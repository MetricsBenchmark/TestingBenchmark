from sklearn.model_selection import train_test_split
import numpy as np
from MNIST_Udacity import mnist
from learner import model_scope_dict
def retrain_cluster(dataset_name, model_name):
    mispred_indices = np.load(f'./result/mispred_indices/{model_name}_misprd_indices.npy')
    cluster_lbls = np.load(f'./result/cluster_lbls/{model_name}_cluster_lbls.npy')
    summary = {}
    if 'mnist' in dataset_name:
        model = mnist.get_model(model_name)
        testX, testy = mnist.get_data('lenet1')  # get_mnist
        testy = onehot_to_int(testy)

    elif 'imdb' in dataset_name:
        testX = np.load(f'./data/tokenized/testX.npy')  # (25000,200)
        testy = np.load(f'./data/tokenized/testy.npy')  # (25000)
        testy = onehot_to_int(testy)
        mispred_indices = np.load(f'./data/tokenized/{model_name}_mispred_idx.npy')
        model = imdb.lm(name=model_name, mode='tokenized')

    elif 'udacity' in dataset_name:
        targeted_model_names_dict = model_scope_dict.copy()
        model = targeted_model_names_dict[model_name](hyper_params=hyper_param, mode='test')

    elif 'andro' in dataset_name:
        targeted_model_names_dict = model_scope_dict.copy()
        model = targeted_model_names_dict[model](mode='test')

    X_mispred, y_mispred = testX[mispred_indices], testy[mispred_indices]
    for cluster_id in np.unique(cluster_lbls):
        if np.sum(cluster_lbls == cluster_id) < 5 or cluster_id == -1:
            continue
        # Select the cluster of interest
        cluster_i = cluster_id
        cluster_mask = (cluster_lbls == cluster_i)
        other_mask = (cluster_lbls != cluster_i) & (cluster_lbls != -1)

        X_cluster = X_mispred[cluster_mask]
        y_cluster = y_mispred[cluster_mask]

        out = [list(y_cluster).count(i) for i in set(list(y_cluster))]
        if 1 in out:
            continue
        X_other = X_mispred[other_mask]
        y_other = y_mispred[other_mask]
        # Split cluster i into 85% train / 15% test
        X_train, X_val, y_train, y_val = train_test_split(X_cluster, y_cluster, test_size=0.15, random_state=42,
                                                          stratify=y_cluster)

        # Retrain the model using 85% of cluster i
        print("Retraining model on cluster {0} (train samples: {1}, val samples: {2})".format(cluster_i, len(X_train),
                                                                                              len(X_val)))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X_train, int_to_onehot(y_train), epochs=3, batch_size=128, verbose=0)
        retrain_pred = model.predict(X_val, verbose=0)
        retrain_pred_int = np.argmax(retrain_pred, axis=1)
        acc_incluster = np.mean(retrain_pred_int == y_val)

        # Evaluate on 15% of cluster i (in-cluster test)
        # Evaluate on all samples from other clusters
        retrain_pred = model.predict(X_other, verbose=0)
        retrain_pred_int = np.argmax(retrain_pred, axis=1)
        acc_other = np.mean(retrain_pred_int == y_other)

        print("In-cluster test accuracy (cluster {0}): {1}".format(cluster_i, acc_incluster),
              "Accuracy on other clusters: {0}".format(acc_other))

        summary[cluster_i] = [len(X_train), acc_incluster, acc_other]
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST experiments")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (only needed for retrain_cluster)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Model name (only needed for retrain_cluster)"
    )
    retrain_cluster(args.dataset, args.model)