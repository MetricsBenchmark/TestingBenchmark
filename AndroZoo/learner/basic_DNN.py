"""
DNN for android malware detection, using the variant of drebin features.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from datetime import datetime
from timeit import default_timer
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.stats as stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from learner.classification import *
from tools import utils
from learner.feature_extractor import get_droid_feature, FeatureMapping, DREBIN_FEAT_INFO, feature_type_scope_dict
from config import config, logging
import copy

logger = logging.getLogger("learning.deepdrebin")

INFO = {
    'dataset_dir': config.get('dataset', 'dataset_root'),
    'feature_type': list(feature_type_scope_dict.keys())[0],  # 'drebin',
    'feature_mapping_type': config.get('feature.drebin', 'feature_mp'),
    'use_interdependent_features': False,
    'learning_algorithm': 'DNN'
}

model_name = 'BASIC_DNN'

DEEPDREBIN_LABEL = 'BASIC_DNN'
BASIC_DNN_LABEL = 'B_DNN'

hidden_units = model_name == DEEPDREBIN_LABEL and [200, 200] or [160, 160]
# print('*********************hidden_units*********************', hidden_units)
# default: 20
TEST_BATCH_SIZE = 20 * 50
# default: 128
BATCH_SIZE = 128 #128 #* 64

DNN_HP = {
    'random_seed': 23456,
    'hidden_units': hidden_units,
    'output_dim': 2,  # malicious vs. benign
    'n_epochs': 150,
    'batch_size': BATCH_SIZE,
    'learning_rate': 0.1,#0.001,
    'optimizer': 'adam'  # others are not supported
}

# DNN_HP = {
#     'random_seed': 23456,
#     'hidden_units': [200, 200],  # DNN has two hidden layers with each having 160 neurons
#     'output_dim': 2,  # malicious vs. benign
#     'n_epochs': 150,
#     'batch_size': BATCH_SIZE,
#     'learning_rate': 0.001,
#     'optimizer': 'adam'  # others are not supported
# }

# def graph(x_input,
#           hidden_neurons=[200, 200],
#           output_dim=2,
#           is_training=True,
#           name="Deepdrebin",
#           reuse=False):
#     '''
#     define the architecture of nerual network
#     :param x_input: data 2D tensor
#     :param hidden_neurons: neurons for hidden layers
#     :param output_dim: number of classes
#     :param is_training: option for training
#     :param name: graph name
#     :param reuse: option for reusing variables
#     :return: graph
#     '''
#     with tf.variable_scope("{}".format(name), reuse=reuse):
#         # dense layer #1 ~ #len(hidden_layers)
#         # the neuron unit is layer_neurons[0]
#         # input Tensor shape: [batch_size, input_dim]
#         # output Tensor shape:[batch_size, hidden_neurons[0]]
#         dense1 = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu, name="DENSE1")

#         #dense1 = tf.layers.dropout(dense1,rate=0.4,training=True,name="DROPOUT1")
#         # dense layer #2
#         # the neuron unit is layer_neurons[1]
#         # input Tensor shape: [batch_size, hidden_neurons[0]]
#         # output Tensor shape:[batch_size, hidden_neurons[1]]
#         dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[1], activation=tf.nn.relu, name="DENSE2")
#         #dense2 = tf.layers.dropout(dense2,rate=0.4,training=True,name="DROPOUT2")
#         # bottlenect
#         # dense layer #3
#         # the neron unit output_dim
#         # input Tensor shape: [batch_size, hidden_neurons[1]]
#         # output Tensor shape: [batch_size, output_dim]
#         dense3 = tf.layers.dense(inputs=dense2, units=output_dim, activation=None, name="DENSE3")

#         return dense1, dense2, dense3

def graph(x_input,
          hidden_neurons=[160, 160],
          output_dim=2,
          is_training=True,
          name=model_name,
          reuse=False):
    '''
    define the architecture of nerual network
    :param x_input: data 2D tensor
    :param hidden_neurons: neurons for hidden layers
    :param output_dim: number of classes
    :param is_training: option for training
    :param name: graph name
    :param reuse: option for reusing variables
    :return: graph 
    '''
    with tf.variable_scope("{}".format(name), reuse=tf.AUTO_REUSE):
        # dense layer #1 ~ #len(hidden_layers)
        # the neuron unit is layer_neurons[0]
        # input Tensor shape: [batch_size, input_dim]
        # output Tensor shape:[batch_size, hidden_neurons[0]]
        dense1 = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu, name="DENSE1")

        # dense layer #2
        # the neuron unit is layer_neurons[1]
        # input Tensor shape: [batch_size, hidden_neurons[0]]
        # output Tensor shape:[batch_size, hidden_neurons[1]]
        dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[1], activation=tf.nn.relu, name="DENSE2")

        # bottlenect
        # dense layer #3
        # the neron unit output_dim
        # input Tensor shape: [batch_size, hidden_neurons[1]]
        # output Tensor shape: [batch_size, output_dim]
        dense3 = tf.layers.dense(inputs=dense2, units=output_dim, activation=None, name="DENSE3")

        return dense1, dense2, dense3


def tester(sess, testX, testy, model, feature=False, required_info='label'):
    """
    model testing on test dataset
    :param sess: tf.Session
    :param testX: data for testing, type: 2-D float np.ndarry
    :param testy: corresponding ground truth labels, type: 1-D int np.ndarray
    :param model: trained model
    :params required_info: 'label' or 'prob'
    :return: predicted label or probability
    """
    test_input = utils.DataProducer(testX, testy, batch_size=480, name='test')
    if isinstance(testy, np.ndarray):
        test_num = testy.shape[0]
    else:
        test_num = len(testy)

    # check
    if required_info in 'label':
        info = model.y_pred
    elif required_info in 'proba':
        info = model.y_proba
    else:
        raise ValueError("'label' or 'proba' is supported.")

    with sess.as_default():
        pred, feat = [], []
        for _, x, y in test_input.next_batch():
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            _y_pred = sess.run(info, feed_dict=test_dict)
            pred.append(_y_pred)
            if feature == True:
                _feat = sess.run(model.dense1, feed_dict=test_dict)
                feat.append(_feat)
        if feature == True:
            return np.concatenate(pred)[:test_num], np.concatenate(feat)
        else:
            return np.concatenate(pred)[:test_num]


def macro_f1_socre(tpr, pos_count, tnr, neg_count):
    """
    macro f1 score for multiple classification
    """
    fnr = 1. - tpr
    fpr = 1. - tnr
    tp = np.rint(pos_count * tpr)
    tn = np.rint(neg_count * tnr)
    fp = np.rint(neg_count * fpr)
    fn = np.rint(pos_count * fnr)

    def binary_f1(tp, fp, fn):
        prec_pos = tp / (tp + fp)
        recall_pos = tp / (tp + fn)
        return 2 * (prec_pos * recall_pos) / (prec_pos + recall_pos)

    f1_pos = binary_f1(tp, fp, fn)
    f1_neg = binary_f1(tn, fn, fp)
    macro_f1 = (f1_pos + f1_neg) / 2.
    return macro_f1

## The DeepGini score increases with the classification uncertainty
def deepgini(sess, candidateX, candidatey, model, budget):
    test_input = utils.DataProducer(candidateX, candidatey, batch_size=120, name='test')
    gini = 1 - tf.reduce_sum(model.y_proba ** 2, axis=1)
    with sess.as_default():
        score = np.array([0.0])
        for _, x, y in test_input.next_batch(): # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }

            _gini_score = sess.run(gini, feed_dict=test_dict)
            score = np.hstack((score, _gini_score.squeeze()))

            #score.append(_gini_score.squeeze())

    scores = np.array(score[1:]).flatten()
    # select ratio from the largest scores
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = candidateX[selected_idx]
    selected_candidatey = candidatey[selected_idx]
    return selected_candidateX, selected_candidatey, scores  # Gini scores for all test inputs


def entropy(sess, candidateX, candidatey, model, budget):
    test_input = utils.DataProducer(candidateX, candidatey, batch_size=120, name='test')
    entropies = -1 * tf.reduce_sum(model.y_proba * tf.log(tf.clip_by_value(model.y_proba, 1e-8, 1.0)), axis=1) # we have clipped the prob to min 1e-8 to avoid Nan when prob=0
    with sess.as_default():
        score = np.array([0.0])
        for _, x, y in test_input.next_batch():  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }

            _ent_score = sess.run(entropies, feed_dict=test_dict)
            score = np.hstack((score, _ent_score.squeeze()))

    scores = np.array(score[1:]).flatten()
    # select ratio from the largest scores
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = candidateX[selected_idx]
    selected_candidatey = candidatey[selected_idx]
    return selected_candidateX, selected_candidatey, scores # entropy scores for all test inputs

def Random(sess, candidateX, candidatey, budget):
    selection_size = int(budget * candidateX.shape[0])
    select_idx = np.random.choice(np.arange(len(candidateX)), selection_size)
    selected_candidateX, selected_candidatey = candidateX[select_idx], candidatey[select_idx]

    return selected_candidateX, selected_candidatey

from sklearn import preprocessing
def GD(sess, candidateX, candidatey, model, selection_size, number=60):
    '''
    Black-box diversity test selection metric

    Step 1: Feature Extraction. Extract features for each sample in the candidateX -> Shape (len(canX), m)
    Step 2: Randomly select with replacement N=number groups of size n=budget*len(idx) from the feature matrix.
    Step 3: For each group -> Nomralize column-wise (feature-wise)
    Step 4: Calculate diversity score GD (geometric diversity)
    Step 5: Select the group with the highest GD.
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    test_input = utils.DataProducer(candidateX, candidatey, batch_size=120, name='test')
    #selection_size = int(budget * candidateX.shape[0])

    if model.model_name.upper() == DEEPDREBIN_LABEL:
        hidden_size = 200
    elif model.model_name.upper() == BASIC_DNN_LABEL:
        hidden_size = 160
    else:
        raise ValueError("Model name not supported.")

    with sess.as_default():
        feat = np.zeros((1,hidden_size))
        for _, x, y in test_input.next_batch():  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }

            _feat = sess.run(model.dense2, feed_dict=test_dict)
            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        GD_scores, selected_indices = [], []
        for _ in range(number):
            select_idx = np.random.choice(np.arange(len(feat_mat)), selection_size)
            selected_indices.append(select_idx)
            select_group = feat_mat[select_idx] #shape (bg*len, feat)
            # normalize group
            normalize_select_group = min_max_scaler.fit_transform(select_group)
            # compute GD
            GD = np.linalg.det(np.matmul(normalize_select_group, normalize_select_group.T))
            GD_scores.append(GD.squeeze())

        max_idx = np.argmax(np.array(GD_scores))
        chosen_indices = selected_indices[max_idx]
        selected_candidateX, selected_candidatey = candidateX[chosen_indices], candidatey[chosen_indices]

    return selected_candidateX, selected_candidatey

def compute_gd(sess, candidateX, candidatey, model):

    min_max_scaler = preprocessing.MinMaxScaler()
    test_input = utils.DataProducer(candidateX, candidatey, batch_size=1, name='test')

    if model.model_name.upper() == DEEPDREBIN_LABEL:
        hidden_size = 200
    elif model.model_name.upper() == BASIC_DNN_LABEL:
        hidden_size = 160
    else:
        raise ValueError("Model name not supported.")

    with sess.as_default():
        feat = np.zeros((1, hidden_size))
        for _, x, y in test_input.next_batch():  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            _feat = sess.run(model.dense2, feed_dict=test_dict)
            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        # normalize group
        normalize_select_group = min_max_scaler.fit_transform(feat_mat)
        # compute GD
        GD = np.linalg.det(np.matmul(normalize_select_group, normalize_select_group.T))
    return GD

def dat(sess, candidateX, candidatey, candidateX_id, candidatey_id, hybrid_test, hybrid_testy, model, id_ratio, budget):
    select_size = budget * candidateX.shape[0]
    id_select_num = int(select_size * id_ratio)    #e.g., 4 out of 40
    ood_select_num = int(select_size - id_select_num)
    para_there = 0.5 # general default value as specified in the paper.
    # Decide how many ID and OOD data we need
    tot_ood_size = int(candidateX.shape[0] - candidateX_id.shape[0])
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)
        else:
            flag_num = id_select_num
            id_select_num = ood_select_num
            ood_select_num = int(flag_num)
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = int(select_size - id_select_num)
        if ood_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)

    print("id num: ", id_select_num)
    print("ood num: ", ood_select_num)

    # select ID by DeepGini
    _,_, id_scores = deepgini(sess, candidateX_id, candidatey_id, model, budget)
    # select ratio from the largest scores
    idx = np.argsort(id_scores)[::-1]
    id_selected_idx = idx[:int(id_select_num)]
    selected_candidateX_id = candidateX_id[id_selected_idx]
    selected_candidatey_id = candidatey_id[id_selected_idx]

    # ood data selection
    # top 1 label
    candidate_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: candidateX, model.y_input: candidatey, model.is_training: False})  #np.argmax(model.predict(candidateX), axis=1)
    reference_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: hybrid_test, model.y_input: hybrid_testy, model.is_training: False}) #np.argmax(model.predict(hybrid_test), axis=1)
    # print(reference_prediction_label)
    reference_labels = []
    for i in range(0, 2):
        label_num = len(np.where(reference_prediction_label == i)[0])
        # print("label {}, num {}".format(i, label_num))
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(candidateX) / select_size
    reference_labels = reference_labels / s_ratio

    ood_part_index = np.arange(candidateX.shape[0])[candidateX_id.shape[0]:]

    label_list = []
    index_list = []

    for _ in range(1000):
        ood_select_index = np.random.choice(ood_part_index, ood_select_num, replace=False)

        this_labels = candidate_prediction_label[ood_select_index.astype(np.int)]
        single_labels = []
        for i in range(0, 2):
            label_num = len(np.where(this_labels == i)[0])
            single_labels.append(label_num)
        index_list.append(ood_select_index)
        label_list.append(single_labels)
    index_list = np.asarray(index_list)
    label_list = np.asarray(label_list)

    # compare to test label
    label_minus = np.abs(label_list - reference_labels)
    var_list = np.sum(label_minus, axis=1)
    var_list_order = np.argsort(var_list)

    ood_select_index = index_list[var_list_order[0]]
    selected_candidateX_ood = candidateX[ood_select_index]
    selected_candidatey_ood = candidatey[ood_select_index]
    selected_data = np.concatenate((selected_candidateX_id, selected_candidateX_ood), axis=0)
    selected_label = np.concatenate((selected_candidatey_id, selected_candidatey_ood), axis=0)

    return selected_data, selected_label # selected candidate data for retraining

def dat_ood_detector(sess, trainX, trainy, candidateX, candidatey, hybrid_test, hybrid_testy, model, budget):
    '''
       introduce OOD detector: using the first hidden-layer feature
       '''
    if model.model_name.upper() == DEEPDREBIN_LABEL:
        hidden_size = 200
    elif model.model_name.upper() == BASIC_DNN_LABEL:
        hidden_size = 160
    else:
        raise ValueError("Model name not supported.")

    train_input = utils.DataProducer(trainX, trainy, batch_size=TEST_BATCH_SIZE, name='test')
    can_input = utils.DataProducer(candidateX, candidatey, batch_size=TEST_BATCH_SIZE, name='test')
    with sess.as_default():
        feat = np.zeros((1, hidden_size))
        feat_can = np.zeros((1, hidden_size))
        for _, x, y in train_input.next_batch():  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            # we select the shallow layer as it is more relevant/representative to the input data instead of the model.
            _feat = sess.run(model.dense1, feed_dict=test_dict)
            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        for _, x, y in can_input.next_batch():  # not shuffle
            can_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            # we select the shallow layer as it is more relevant/representative to the input data instead of the model.
            _feat = sess.run(model.dense1, feed_dict=can_dict)
            feat_can = np.vstack((feat_can, _feat))
        feat_mat_can = feat_can[1:]

    # Use train set to compute center of the hypersphere, it is defined as the mean feature of training data (ID data)
    center = np.mean(feat_mat, axis=0)
    # compute distance of each train feature to center
    dist = [np.sum((i - center) ** 2) for i in feat_mat]
    dist_sort = np.sort(dist)  # from smallest to biggest
    threshold = dist_sort[
        int(0.95 * (len(dist_sort)))]  # select threshold such that 95% train data are correctly classified as ID

    dist_can = [np.sum((i - center) ** 2) for i in feat_mat_can]
    candidateX_id, candidateX_ood = candidateX[dist_can <= threshold], candidateX[dist_can > threshold]
    candidatey_id, candidatey_ood = candidatey[dist_can <= threshold], candidatey[dist_can > threshold]


    select_size = budget * candidateX.shape[0]
    id_select_num = int(candidateX_id.shape[0] * budget)  # int(select_size * id_ratio)    #e.g., 4 out of 40
    ood_select_num = int(candidateX_ood.shape[0] * budget)  # int(select_size - id_select_num)
    para_there = 0.5 # general default value as specified in the paper.
    # Decide how many ID and OOD data we need
    tot_ood_size = int(candidateX_ood.shape[0])
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)
        else:
            flag_num = id_select_num
            id_select_num = ood_select_num
            ood_select_num = int(flag_num)
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = int(select_size - id_select_num)
        if ood_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)

    print("id num: ", id_select_num)
    print("ood num: ", ood_select_num)

    # select ID by DeepGini
    _,_, id_scores = deepgini(sess, candidateX_id, candidatey_id, model, budget)
    # select ratio from the largest scores
    idx = np.argsort(id_scores)[::-1]
    id_selected_idx = idx[:int(id_select_num)]
    selected_candidateX_id = candidateX_id[id_selected_idx]
    selected_candidatey_id = candidatey_id[id_selected_idx]

    # ood data selection
    # top 1 label
    candidate_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: candidateX, model.y_input: candidatey, model.is_training: False})  #np.argmax(model.predict(candidateX), axis=1)
    reference_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: hybrid_test, model.y_input: hybrid_testy, model.is_training: False}) #np.argmax(model.predict(hybrid_test), axis=1)
    # print(reference_prediction_label)
    reference_labels = []
    for i in range(0, 2):
        label_num = len(np.where(reference_prediction_label == i)[0])
        # print("label {}, num {}".format(i, label_num))
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(candidateX) / select_size
    reference_labels = reference_labels / s_ratio

    ood_part_index = np.where((dist_can > threshold) == True)[0]

    label_list = []
    index_list = []
    if ood_select_num != 0:
        for _ in range(1000):
            ood_select_index = np.random.choice(ood_part_index, ood_select_num, replace=False)

            this_labels = candidate_prediction_label[ood_select_index.astype(np.int)]
            single_labels = []
            for i in range(0, 2):
                label_num = len(np.where(this_labels == i)[0])
                single_labels.append(label_num)
            index_list.append(ood_select_index)
            label_list.append(single_labels)
        index_list = np.asarray(index_list)
        label_list = np.asarray(label_list)

        # compare to test label
        label_minus = np.abs(label_list - reference_labels)
        var_list = np.sum(label_minus, axis=1)
        var_list_order = np.argsort(var_list)

        ood_select_index = index_list[var_list_order[0]]
        selected_candidateX_ood = candidateX[ood_select_index]
        selected_candidatey_ood = candidatey[ood_select_index]
        selected_data = np.concatenate((selected_candidateX_id, selected_candidateX_ood), axis=0)
        selected_label = np.concatenate((selected_candidatey_id, selected_candidatey_ood), axis=0)
    else:
        selected_data = selected_candidateX_id
        selected_label = selected_candidatey_id
    return selected_data, selected_label # selected candidate data for retraining

def ensemble_gt_ood(sess, trainX, trainy, candidateX, candidatey, candidateX_id, candidatey_id, hybrid_test, hybrid_testy, model, id_ratio, budget):
    '''
    original "ensemble"
    this uses ground truth label of OOD, not applicable to our unsupervised learning setting
    '''
    select_size = budget * candidateX.shape[0]
    id_select_num = int(select_size * id_ratio)    #e.g., 4 out of 40
    ood_select_num = int(select_size - id_select_num)
    para_there = 0.5 # general default value as specified in the paper.
    # Decide how many ID and OOD data we need
    tot_ood_size = int(candidateX.shape[0] - candidateX_id.shape[0])
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)
        else:
            flag_num = id_select_num
            id_select_num = ood_select_num
            ood_select_num = int(flag_num)
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = int(select_size - id_select_num)
        if ood_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)

    print("id num: ", id_select_num)
    print("ood num: ", ood_select_num)

    _,_, id_scores_gini = deepgini(sess, candidateX_id, candidatey_id, model, budget)
    _,_, train_scores_gini = deepgini(sess, trainX, trainy, model, budget)
    _, _, train_scores_entropy = entropy(sess, trainX, trainy, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX_id, candidatey_id, model, budget)

    # select ratio from the largest scores
    mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    mean_p_values = np.mean(np.vstack((p_values_gini, p_values_entropy)), axis=0)
    idx = np.argsort(mean_p_values) # from smallest (most uncertain) to biggest
    half_num = int(0.5 * len(idx))
    # GD: select most diverse set from the uncertain pool
    print('id_select_num', id_select_num, 'half_num', half_num)
    if id_select_num == 0:
        id_selected_idx = idx[:int(id_select_num)]
        selected_candidateX_id = candidateX_id[id_selected_idx]
        selected_candidatey_id = candidatey_id[id_selected_idx]
    elif half_num > id_select_num:
        selected_candidateX_id, selected_candidatey_id = GD(sess, candidateX_id[idx[:half_num]], candidatey_id[idx[:half_num]], model, id_select_num, number=60)
    else:
        selected_candidateX_id, selected_candidatey_id = GD(sess, candidateX_id,
                                                            candidatey_id, model, id_select_num,
                                                            number=60)

    #id_selected_idx = idx[:int(id_select_num)]
    #selected_candidateX_id = candidateX_id[id_selected_idx]
    #selected_candidatey_id = candidatey_id[id_selected_idx]

    # ood data selection
    # top 1 label
    candidate_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: candidateX, model.y_input: candidatey, model.is_training: False})  #np.argmax(model.predict(candidateX), axis=1)
    reference_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: hybrid_test, model.y_input: hybrid_testy, model.is_training: False}) #np.argmax(model.predict(hybrid_test), axis=1)
    # print(reference_prediction_label)
    reference_labels = []
    for i in range(0, 2):
        label_num = len(np.where(reference_prediction_label == i)[0])
        # print("label {}, num {}".format(i, label_num))
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(candidateX) / select_size
    reference_labels = reference_labels / s_ratio

    ood_part_index = np.arange(candidateX.shape[0])[candidateX_id.shape[0]:]

    label_list = []
    index_list = []

    for _ in range(1000):
        ood_select_index = np.random.choice(ood_part_index, ood_select_num, replace=False)

        this_labels = candidate_prediction_label[ood_select_index.astype(np.int)]
        single_labels = []
        for i in range(0, 2):
            label_num = len(np.where(this_labels == i)[0])
            single_labels.append(label_num)
        index_list.append(ood_select_index)
        label_list.append(single_labels)
    index_list = np.asarray(index_list)
    label_list = np.asarray(label_list)

    # compare to test label & todo: choose the one with the highest diversity?
    label_minus = np.abs(label_list - reference_labels)
    var_list = np.sum(label_minus, axis=1)
    var_list_order = np.argsort(var_list)

    ood_select_index = index_list[var_list_order[0]]
    selected_candidateX_ood = candidateX[ood_select_index]
    selected_candidatey_ood = candidatey[ood_select_index]
    selected_data = np.concatenate((selected_candidateX_id, selected_candidateX_ood), axis=0)
    selected_label = np.concatenate((selected_candidatey_id, selected_candidatey_ood), axis=0)
    return selected_data, selected_label # selected candidate data for retraining

def ensemble_p_values(sess, trainX, trainy, candidateX, candidatey, model, budget, our_method_params):
    '''
    Use p-values.
    Does not utilize OOD detector to distinguish ID & OOD
    just use GD to select from uncertain batch (by p-values)
    '''
    w_gini, w_ent, num, ratio = our_method_params['w_gini'], our_method_params['w_ent'], our_method_params['gd_num'], our_method_params['ratio']
    selection_size = int(budget * candidateX.shape[0])

    _,_, id_scores_gini = deepgini(sess, candidateX, candidatey, model, budget)
    _,_, train_scores_gini = deepgini(sess, trainX, trainy, model, budget)
    _, _, train_scores_entropy = entropy(sess, trainX, trainy, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)

    # select ratio from the largest scores
    mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    weighted_p_vals = w_gini * p_values_gini + w_ent * p_values_entropy
    idx = np.argsort(weighted_p_vals)  # from smallest (most uncertain) to biggest
    half_num = int(ratio * len(idx))
    # GD: select most diverse set from the uncertain pool

    if half_num > selection_size:
        selected_candidateX, selected_candidatey = GD(sess, candidateX[idx[:half_num]], candidatey[idx[:half_num]], model, selection_size, number=num)
    else:
        selected_candidateX, selected_candidatey = GD(sess, candidateX,
                                                            candidatey, model, selection_size,
                                                            number=num)
    #id_selected_idx = idx[:int(id_select_num)]
    #selected_candidateX_id = candidateX_id[id_selected_idx]
    #selected_candidatey_id = candidatey_id[id_selected_idx]

    return selected_candidateX, selected_candidatey # selected candidate data for retraining

def ensemble_p_values_incre_change(sess, trainX, trainy, candidateX, candidatey, model, budget, our_method_params):
    '''
    Use p-values for input-level metrics & incremental change for suite-level metrics
    Does not utilize OOD detector to distinguish ID & OOD
    '''
    bs, w_gini, w_ent = our_method_params['batch_size_ensemble'], our_method_params['w_gini'], our_method_params['w_ent']
    selection_size = int(budget * candidateX.shape[0])

    _,_, id_scores_gini = deepgini(sess, candidateX, candidatey, model, budget)
    _,_, train_scores_gini = deepgini(sess, trainX, trainy, model, budget)
    _, _, train_scores_entropy = entropy(sess, trainX, trainy, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)

    # select ratio from the largest scores
    mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    weighted_p_vals = w_gini * p_values_gini + w_ent * p_values_entropy
    idx = np.argsort(weighted_p_vals) #from smallest (most uncertain) to biggest
    prev_gd = 0.0
    size = 1
    pool, pool_y, rest_idx = [], [], []
    # random initialization
    pool.append(candidateX[idx[0]])
    pool_y.append(candidatey[idx[0]])
    good_pool = copy.deepcopy(pool)
    good_pool_y = copy.deepcopy(pool_y)

    for i in np.arange(1, len(idx)):
        add_sample = candidateX[idx[i]]
        add_y = candidatey[idx[i]]
        pool.append(add_sample)
        pool_y.append(add_y)
        # incremental change
        # todo: plot actual change in GD when adding new sample
        current_gd = compute_gd(sess, np.array(pool), np.array(pool_y), model)
        if current_gd > prev_gd:
            good_pool.append(add_sample)
            good_pool_y.append(add_y)
            size = len(good_pool_y)
            prev_gd = current_gd
        else:
            rest_idx.append(i)
        pool, pool_y = copy.deepcopy(good_pool), copy.deepcopy(good_pool_y)

    if size < selection_size:
        nest_size = selection_size - size
        indices = idx[rest_idx[:nest_size]] # fill in the rest with the most uncertain samples
        good_pool = np.vstack((np.array(good_pool), candidateX[indices]))
        good_pool_y = np.hstack((np.array(good_pool_y), candidatey[indices]))

    selected_candidateX, selected_candidatey = np.array(good_pool), np.array(good_pool_y)
    return selected_candidateX, selected_candidatey # selected candidate data for retraining

def ensemble_algorithm_3(sess, trainX, trainy, candidateX, candidatey, model, budget):
    '''
    Use p-values for input-level metrics & incremental change for suite-level metrics
    Does not utilize OOD detector to distinguish ID & OOD
    '''
    selection_size = int(budget * candidateX.shape[0])
    # get gini scores for each test input.
    _,_, gini_scores = deepgini(sess, candidateX, candidatey, model, budget) # the bigger the more uncertain
    # select ratio from the largest scores
    idx = np.argsort(gini_scores)[::-1]
    size = 1
    # we initialize with the most uncertain sample
    good_pool, good_pool_y = [], []
    good_pool.append(candidateX[idx[0]])
    good_pool_y.append(candidatey[idx[0]])
    start = time.time()
    restX, resty = copy.deepcopy(np.delete(candidateX, idx[0], 0)), copy.deepcopy(np.delete(candidatey, idx[0]))
    rest_gini_scores = copy.deepcopy(np.delete(gini_scores, idx[0]))
    while size < selection_size:
        # compute gd scores for all samples (todo: this is way too slow!)
        gd_scores = []
        for i in np.arange(0, len(resty)):
            gd = compute_gd(sess, np.vstack((good_pool, restX[i])), np.hstack((good_pool_y, resty[i])), model)
            gd_scores.append(gd)

        combined_score = rest_gini_scores + np.array(gd_scores)
        idx = np.argsort(combined_score)[::-1]
        good_pool.append(restX[idx[0]])
        good_pool_y.append(resty[idx[0]])
        restX, resty = copy.deepcopy(np.delete(restX, idx[0], 0)), copy.deepcopy(np.delete(resty, idx[0]))
        rest_gini_scores = copy.deepcopy(np.delete(rest_gini_scores, idx[0]))
        size = len(good_pool_y)
    print('time cost: ', time.time() - start)
    selected_candidateX, selected_candidatey = np.array(good_pool), np.array(good_pool_y)
    return selected_candidateX, selected_candidatey # selected candidate data for retraining

def ensemble_algorithm_3_batch(sess, candidateX, candidatey, model, budget, our_method_params):
    '''
    Use p-values for input-level metrics & incremental change for suite-level metrics
    Does not utilize OOD detector to distinguish ID & OOD
    '''
    bs, w_gini, w_gd = our_method_params['batch_size_ensemble'], our_method_params['w_gini'], our_method_params['w_gd']
    selection_size = int(budget * candidateX.shape[0])
    # get gini scores for each test input.
    _,_, gini_scores = deepgini(sess, candidateX, candidatey, model, budget) # the bigger the more uncertain
    # select ratio from the largest scores
    idx = np.argsort(gini_scores)[::-1]
    size = 1
    # we initialize with the most uncertain sample
    good_pool, good_pool_y = [], []
    good_pool.append(candidateX[idx[0]])
    good_pool_y.append(candidatey[idx[0]])
    start = time.time()
    restX, resty = copy.deepcopy(np.delete(candidateX, idx[0], 0)), copy.deepcopy(np.delete(candidatey, idx[0]))
    rest_gini_scores = copy.deepcopy(np.delete(gini_scores, idx[0]))
    while size < selection_size:
        # compute gd scores for all samples (todo: this is way too slow!)
        gd_scores = []
        for i in np.arange(0, len(resty)):
            gd = compute_gd(sess, np.vstack((good_pool, restX[i])), np.hstack((good_pool_y, resty[i])), model)
            gd_scores.append(gd)
        print('finish gd_scores appending')
        combined_score = w_gini * rest_gini_scores + w_gd * np.array(gd_scores) #todo: add entropy, record three wieghts
        print('1')
        idx = np.argsort(combined_score)[::-1][:bs]
        print(2)
        good_pool = np.vstack((good_pool, restX[idx]))
        good_pool_y = np.hstack((good_pool_y, resty[idx]))
        print(3)
        restX, resty = copy.deepcopy(np.delete(restX, idx, 0)), copy.deepcopy(np.delete(resty, idx))
        print(4)
        rest_gini_scores = copy.deepcopy(np.delete(rest_gini_scores, idx))
        print(5)
        size = len(good_pool_y)
        print(size, selection_size, restX.shape)

    print('time cost: ', time.time() - start)
    selected_candidateX, selected_candidatey = good_pool[:selection_size], good_pool_y[:selection_size]
    print('selected size', selected_candidateX.shape, selected_candidatey.shape, selection_size)
    return selected_candidateX, selected_candidatey # selected candidate data for retraining

def ensemble_simple(sess, candidateX, candidatey, model, budget):
    '''
    No p-values.
    Does not utilize OOD detector to distinguish ID & OOD
    just use GD to select from uncertain batch by Entropy
    '''

    selection_size = int(budget * candidateX.shape[0])

    _,_, id_scores_gini = deepgini(sess, candidateX, candidatey, model, budget)
    #_,_, train_scores_gini = deepgini(sess, trainX, trainy, model, budget)
    #_, _, train_scores_entropy = entropy(sess, trainX, trainy, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)

    # select ratio from the largest scores
    #mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    #mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    #p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    #p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    #mean_p_values = np.mean(np.vstack((p_values_gini, p_values_entropy)), axis=0)
    #idx = np.argsort(mean_p_values) # from smallest (most uncertain) to biggest
    idx = np.argsort(id_scores_gini)[::-1]
    half_num = int(0.5 * len(idx))
    # GD: select most diverse set from the uncertain pool

    if half_num > selection_size:
        selected_candidateX, selected_candidatey = GD(sess, candidateX[idx[:half_num]], candidatey[idx[:half_num]], model, selection_size, number=60)
    else:
        selected_candidateX, selected_candidatey = GD(sess, candidateX,
                                                            candidatey, model, selection_size,
                                                            number=60)
    #id_selected_idx = idx[:int(id_select_num)]
    #selected_candidateX_id = candidateX_id[id_selected_idx]
    #selected_candidatey_id = candidatey_id[id_selected_idx]

    return selected_candidateX, selected_candidatey # selected candidate data for retraining

def ensemble_simple_add(sess, candidateX, candidatey, model, budget):
    '''
    No p-values.
    Does not utilize OOD detector to distinguish ID & OOD
    just use GD to select from uncertain batch by Entropy
    '''

    selection_size = int(budget * candidateX.shape[0])

    _,_, id_scores_gini = deepgini(sess, candidateX, candidatey, model, budget)
    #_,_, train_scores_gini = deepgini(sess, trainX, trainy, model, budget)
    #_, _, train_scores_entropy = entropy(sess, trainX, trainy, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)

    # select ratio from the largest scores
    #mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    #mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    #p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    #p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    #mean_p_values = np.mean(np.vstack((p_values_gini, p_values_entropy)), axis=0)
    #idx = np.argsort(mean_p_values) # from smallest (most uncertain) to biggest
    idx = np.argsort(id_scores_gini+id_scores_entropy)[::-1]
    half_num = int(0.5 * len(idx))
    # GD: select most diverse set from the uncertain pool
    if half_num > selection_size:
        selected_candidateX, selected_candidatey = GD(sess, candidateX[idx[:half_num]], candidatey[idx[:half_num]], model, selection_size, number=60)
    else:
        selected_candidateX, selected_candidatey = GD(sess, candidateX,
                                                            candidatey, model, selection_size,
                                                            number=60)
    #id_selected_idx = idx[:int(id_select_num)]
    #selected_candidateX_id = candidateX_id[id_selected_idx]
    #selected_candidatey_id = candidatey_id[id_selected_idx]

    return selected_candidateX, selected_candidatey # selected candidate data for retraining

def ensemble_ood_detector(sess, trainX, trainy, candidateX, candidatey, hybrid_test, hybrid_testy, model, budget):
    '''
    introduce OOD detector: using the first hidden-layer feature
    for ID: use GD to select from uncertain pool
    for OOD: the same as DAT
    '''
    if model.model_name.upper() == DEEPDREBIN_LABEL:
        hidden_size = 200
    elif model.model_name.upper() == BASIC_DNN_LABEL:
        hidden_size = 160
    else:
        raise ValueError("Model name not supported.")

    train_input = utils.DataProducer(trainX, trainy, batch_size=480, name='test')
    can_input = utils.DataProducer(candidateX, candidatey, batch_size=480, name='test')
    with sess.as_default():
        feat = np.zeros((1,hidden_size))
        feat_can = np.zeros((1,hidden_size))
        for _, x, y in train_input.next_batch():  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            # we select the shallow layer as it is more relevant/representative to the input data instead of the model.
            _feat = sess.run(model.dense2, feed_dict=test_dict)
            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        for _, x, y in can_input.next_batch():  # not shuffle
            can_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            # we select the shallow layer as it is more relevant/representative to the input data instead of the model.
            _feat = sess.run(model.dense2, feed_dict=can_dict)
            feat_can = np.vstack((feat_can, _feat))
        feat_mat_can = feat_can[1:]

    # Use train set to compute center of the hypersphere, it is defined as the mean feature of training data (ID data)
    center = np.mean(feat_mat, axis=0)
    # compute distance of each train feature to center
    dist = [np.sum((i-center)**2) for i in feat_mat]
    dist_sort = np.sort(dist) # from smallest to biggest
    threshold = dist_sort[int(0.95*(len(dist_sort)))] # select threshold such that 95% train data are correctly classified as ID

    dist_can = [np.sum((i-center)**2) for i in feat_mat_can]
    candidateX_id, candidateX_ood = candidateX[dist_can <= threshold], candidateX[dist_can > threshold]
    candidatey_id, candidatey_ood = candidatey[dist_can <= threshold], candidatey[dist_can > threshold]

    select_size = budget * candidateX.shape[0]
    id_select_num = int(candidateX_id.shape[0] * budget)#int(select_size * id_ratio)    #e.g., 4 out of 40
    ood_select_num = int(candidateX_ood.shape[0] * budget)#int(select_size - id_select_num)
    para_there = 0.5 # general default value as specified in the paper.
    # Decide how many ID and OOD data we need
    tot_ood_size = int(candidateX_ood.shape[0])
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)
        else:
            flag_num = id_select_num
            id_select_num = ood_select_num
            ood_select_num = int(flag_num)
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = int(select_size - id_select_num)
        if ood_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)

    print("id num: ", id_select_num)
    print("ood num: ", ood_select_num)

    _,_, id_scores_gini = deepgini(sess, candidateX_id, candidatey_id, model, budget)
    _,_, train_scores_gini = deepgini(sess, trainX, trainy, model, budget)
    _, _, train_scores_entropy = entropy(sess, trainX, trainy, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX_id, candidatey_id, model, budget)

    # select ratio from the largest scores
    mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    mean_p_values = np.mean(np.vstack((p_values_gini, p_values_entropy)), axis=0)
    idx = np.argsort(mean_p_values) # from smallest (most uncertain) to biggest
    half_num = int(0.5 * len(idx))
    # GD: select most diverse set from the uncertain pool
    print('id_select_num', id_select_num, 'half_num', half_num)
    if id_select_num == 0:
        id_selected_idx = idx[:int(id_select_num)]
        selected_candidateX_id = candidateX_id[id_selected_idx]
        selected_candidatey_id = candidatey_id[id_selected_idx]
    elif half_num > id_select_num:
        selected_candidateX_id, selected_candidatey_id = GD(sess, candidateX_id[idx[:half_num]], candidatey_id[idx[:half_num]], model, id_select_num, number=60)
    else:
        selected_candidateX_id, selected_candidatey_id = GD(sess, candidateX_id,
                                                            candidatey_id, model, id_select_num,
                                                            number=60)

    #id_selected_idx = idx[:int(id_select_num)]
    #selected_candidateX_id = candidateX_id[id_selected_idx]
    #selected_candidatey_id = candidatey_id[id_selected_idx]

    # ood data selection
    # top 1 label
    candidate_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: candidateX, model.y_input: candidatey, model.is_training: False})  #np.argmax(model.predict(candidateX), axis=1)
    reference_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: hybrid_test, model.y_input: hybrid_testy, model.is_training: False}) #np.argmax(model.predict(hybrid_test), axis=1)
    # print(reference_prediction_label)
    reference_labels = []
    for i in range(0, 2):
        label_num = len(np.where(reference_prediction_label == i)[0])
        # print("label {}, num {}".format(i, label_num))
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(candidateX) / select_size
    reference_labels = reference_labels / s_ratio

    #ood_part_index = np.arange(candidateX.shape[0])[candidateX_id.shape[0]:] #todo: change 2
    ood_part_index = np.where((dist_can > threshold) == True)[0]

    label_list = []
    index_list = []
    if ood_select_num != 0:
        for _ in range(1000):
            ood_select_index = np.random.choice(ood_part_index, ood_select_num, replace=False)

            this_labels = candidate_prediction_label[ood_select_index.astype(np.int)]
            single_labels = []
            for i in range(0, 2):
                label_num = len(np.where(this_labels == i)[0])
                single_labels.append(label_num)
            index_list.append(ood_select_index)
            label_list.append(single_labels)
        index_list = np.asarray(index_list)
        label_list = np.asarray(label_list)

        # compare to test label & todo: choose the one with the highest diversity?
        label_minus = np.abs(label_list - reference_labels)
        var_list = np.sum(label_minus, axis=1)
        var_list_order = np.argsort(var_list)

        ood_select_index = index_list[var_list_order[0]]
        selected_candidateX_ood = candidateX[ood_select_index]
        selected_candidatey_ood = candidatey[ood_select_index]
        selected_data = np.concatenate((selected_candidateX_id, selected_candidateX_ood), axis=0)
        selected_label = np.concatenate((selected_candidatey_id, selected_candidatey_ood), axis=0)
    else:
        selected_data = selected_candidateX_id
        selected_label = selected_candidatey_id
    return selected_data, selected_label # selected candidate data for retraining

def ensemble_ood_uncertain(sess, trainX, trainy, candidateX, candidatey, hybrid_test, hybrid_testy, model, budget):
    '''
    introduce OOD detector: using the first hidden-layer feature
    For ID data: select most diverse set from the uncertain pool
    For OOD data: the same as DAT
    '''
    if model.model_name.upper() == DEEPDREBIN_LABEL:
        hidden_size = 200
    elif model.model_name.upper() == BASIC_DNN_LABEL:
        hidden_size = 160
    else:
        raise ValueError("Model name not supported.")

    train_input = utils.DataProducer(trainX, trainy, batch_size=480, name='test')
    can_input = utils.DataProducer(candidateX, candidatey, batch_size=480, name='test')
    _, _, train_scores_entropy = entropy(sess, trainX, trainy, model, budget)
    _, _, candidate_scores_entropy = entropy(sess, candidateX, candidatey, model, budget)
    train_score_entropy_sort = np.argsort(train_scores_entropy) # smallest to largest score
    # larger scores indicate more uncertain
    threshold = train_score_entropy_sort[
        int(0.95 * (len(train_score_entropy_sort)))]  # select threshold such that 95% train data are correctly classified as ID


    candidateX_id, candidateX_ood = candidateX[candidate_scores_entropy <= threshold], candidateX[candidate_scores_entropy > threshold]
    candidatey_id, candidatey_ood = candidatey[candidate_scores_entropy <= threshold], candidatey[candidate_scores_entropy > threshold]

    select_size = budget * candidateX.shape[0]
    id_select_num = int(candidateX_id.shape[0] * budget)#int(select_size * id_ratio)    #e.g., 4 out of 40
    ood_select_num = int(candidateX_ood.shape[0] * budget)#int(select_size - id_select_num)
    para_there = 0.5 # general default value as specified in the paper.
    # Decide how many ID and OOD data we need
    tot_ood_size = int(candidateX_ood.shape[0])
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)
        else:
            flag_num = id_select_num
            id_select_num = ood_select_num
            ood_select_num = int(flag_num)
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = int(select_size - id_select_num)
        if ood_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)

    print("id num: ", id_select_num)
    print("ood num: ", ood_select_num)

    _,_, id_scores_gini = deepgini(sess, candidateX_id, candidatey_id, model, budget)
    _,_, train_scores_gini = deepgini(sess, trainX, trainy, model, budget)

    _,_, id_scores_entropy = entropy(sess, candidateX_id, candidatey_id, model, budget)

    # select ratio from the largest scores
    mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    mean_p_values = np.mean(np.vstack((p_values_gini, p_values_entropy)), axis=0)
    idx = np.argsort(mean_p_values) # from smallest (most uncertain) to biggest
    half_num = int(0.5 * len(idx))
    # GD: select most diverse set from the uncertain pool
    print('id_select_num', id_select_num, 'half_num', half_num)
    if id_select_num == 0:
        id_selected_idx = idx[:int(id_select_num)]
        selected_candidateX_id = candidateX_id[id_selected_idx]
        selected_candidatey_id = candidatey_id[id_selected_idx]
    elif half_num > id_select_num:
        selected_candidateX_id, selected_candidatey_id = GD(sess, candidateX_id[idx[:half_num]], candidatey_id[idx[:half_num]], model, id_select_num, number=60)
    else:
        selected_candidateX_id, selected_candidatey_id = GD(sess, candidateX_id,
                                                            candidatey_id, model, id_select_num,
                                                            number=60)

    #id_selected_idx = idx[:int(id_select_num)]
    #selected_candidateX_id = candidateX_id[id_selected_idx]
    #selected_candidatey_id = candidatey_id[id_selected_idx]

    # ood data selection
    # top 1 label
    candidate_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: candidateX, model.y_input: candidatey, model.is_training: False})  #np.argmax(model.predict(candidateX), axis=1)
    reference_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: hybrid_test, model.y_input: hybrid_testy, model.is_training: False}) #np.argmax(model.predict(hybrid_test), axis=1)
    # print(reference_prediction_label)
    reference_labels = []
    for i in range(0, 2):
        label_num = len(np.where(reference_prediction_label == i)[0])
        # print("label {}, num {}".format(i, label_num))
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(candidateX) / select_size
    reference_labels = reference_labels / s_ratio

    #ood_part_index = np.arange(candidateX.shape[0])[candidateX_id.shape[0]:] #todo: change 2
    ood_part_index = np.where((candidate_scores_entropy > threshold) == True)[0]

    label_list = []
    index_list = []
    if ood_select_num != 0:
        for _ in range(1000):
            ood_select_index = np.random.choice(ood_part_index, ood_select_num, replace=False)

            this_labels = candidate_prediction_label[ood_select_index.astype(np.int)]
            single_labels = []
            for i in range(0, 2):
                label_num = len(np.where(this_labels == i)[0])
                single_labels.append(label_num)
            index_list.append(ood_select_index)
            label_list.append(single_labels)
        index_list = np.asarray(index_list)
        label_list = np.asarray(label_list)

        # compare to test label & todo: choose the one with the highest diversity?
        label_minus = np.abs(label_list - reference_labels)
        var_list = np.sum(label_minus, axis=1)
        var_list_order = np.argsort(var_list)

        ood_select_index = index_list[var_list_order[0]]
        selected_candidateX_ood = candidateX[ood_select_index]
        selected_candidatey_ood = candidatey[ood_select_index]
        selected_data = np.concatenate((selected_candidateX_id, selected_candidateX_ood), axis=0)
        selected_label = np.concatenate((selected_candidatey_id, selected_candidatey_ood), axis=0)
    else:
        selected_data = selected_candidateX_id
        selected_label = selected_candidatey_id
    return selected_data, selected_label # selected candidate data for retraining

def ensemble_ood_gd(sess, trainX, trainy, candidateX, candidatey, model, budget):
    '''
    introduce OOD detector: using the first hidden-layer feature
    Moreover, for OOD, We use GD to select more diverse from randomly selected batches, instead of more balanced label distribution
    For ID, we use GD to select from the more uncertain batch
    '''
    if model.model_name.upper() == DEEPDREBIN_LABEL:
        hidden_size = 200
    elif model.model_name.upper() == BASIC_DNN_LABEL:
        hidden_size = 160
    else:
        raise ValueError("Model name not supported.")

    train_input = utils.DataProducer(trainX, trainy, batch_size=480, name='test')
    can_input = utils.DataProducer(candidateX, candidatey, batch_size=480, name='test')
    with sess.as_default():
        feat = np.zeros((1,hidden_size))
        feat_can = np.zeros((1,hidden_size))
        for _, x, y in train_input.next_batch():  # not shuffle
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            # we select the shallow layer as it is more relevant/representative to the input data instead of the model.
            _feat = sess.run(model.dense1, feed_dict=test_dict)
            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        for _, x, y in can_input.next_batch():  # not shuffle
            can_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            # we select the shallow layer as it is more relevant/representative to the input data instead of the model.
            _feat = sess.run(model.dense1, feed_dict=can_dict)
            feat_can = np.vstack((feat_can, _feat))
        feat_mat_can = feat_can[1:]

    # Use train set to compute center of the hypersphere, it is defined as the mean feature of training data (ID data)
    center = np.mean(feat_mat, axis=0)
    # compute distance of each train feature to center
    dist = [np.sum((i-center)**2) for i in feat_mat]
    dist_sort = np.sort(dist) # from smallest to biggest
    threshold = dist_sort[int(0.95*(len(dist_sort)))] # select threshold such that 95% train data are correctly classified as ID

    dist_can = [np.sum((i-center)**2) for i in feat_mat_can]
    candidateX_id, candidateX_ood = candidateX[dist_can <= threshold], candidateX[dist_can > threshold]
    candidatey_id, candidatey_ood = candidatey[dist_can <= threshold], candidatey[dist_can > threshold]

    select_size = budget * candidateX.shape[0]
    id_select_num = int(candidateX_id.shape[0] * budget)#int(select_size * id_ratio)    #e.g., 4 out of 40
    ood_select_num = int(candidateX_ood.shape[0] * budget)#int(select_size - id_select_num)
    para_there = 0.5 # general default value as specified in the paper.
    # Decide how many ID and OOD data we need
    tot_ood_size = int(candidateX_ood.shape[0])
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)
        else:
            flag_num = id_select_num
            id_select_num = ood_select_num
            ood_select_num = int(flag_num)
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = int(select_size - id_select_num)
        if ood_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)

    print("id num: ", id_select_num)
    print("ood num: ", ood_select_num)

    _,_, id_scores_gini = deepgini(sess, candidateX_id, candidatey_id, model, budget)
    _,_, train_scores_gini = deepgini(sess, trainX, trainy, model, budget)
    _, _, train_scores_entropy = entropy(sess, trainX, trainy, model, budget)
    _,_, id_scores_entropy = entropy(sess, candidateX_id, candidatey_id, model, budget)

    # select ratio from the largest scores
    mean_gini, std_gini = stats.tmean(train_scores_gini.squeeze()), stats.tstd(train_scores_gini.squeeze())
    mean_entropy, std_entropy = stats.tmean(train_scores_entropy.squeeze()), stats.tstd(train_scores_entropy.squeeze())

    p_values_gini = 1 - stats.norm.cdf(id_scores_gini, mean_gini, std_gini)
    p_values_entropy = 1 - stats.norm.cdf(id_scores_entropy, mean_entropy, std_entropy)

    # mean, Dir?
    mean_p_values = np.mean(np.vstack((p_values_gini, p_values_entropy)), axis=0)
    idx = np.argsort(mean_p_values) # from smallest (most uncertain) to biggest
    half_num = int(0.5 * len(idx))
    # GD: select most diverse set from the uncertain pool
    print('id_select_num', id_select_num, 'half_num', half_num)
    if id_select_num == 0:
        id_selected_idx = idx[:int(id_select_num)]
        selected_candidateX_id = candidateX_id[id_selected_idx]
        selected_candidatey_id = candidatey_id[id_selected_idx]
    elif half_num > id_select_num:
        selected_candidateX_id, selected_candidatey_id = GD(sess, candidateX_id[idx[:half_num]], candidatey_id[idx[:half_num]], model, id_select_num, number=60)
    else:
        selected_candidateX_id, selected_candidatey_id = GD(sess, candidateX_id,
                                                            candidatey_id, model, id_select_num,
                                                            number=60)

    if ood_select_num != 0:
        selected_candidateX_ood, selected_candidatey_ood = GD(sess, candidateX_ood,
                                                            candidatey_ood, model, ood_select_num,
                                                            number=60)
        selected_data = np.concatenate((selected_candidateX_id, selected_candidateX_ood), axis=0)
        selected_label = np.concatenate((selected_candidatey_id, selected_candidatey_ood), axis=0)
    else:
        selected_data = selected_candidateX_id
        selected_label = selected_candidatey_id
    return selected_data, selected_label # selected candidate data for retraining

class BasicDNNModel(Classifier):
    def __init__(self,
                 info_dict=None,
                 hyper_params=None,
                 reuse=False,
                 is_saving=True,
                 init_graph=True,
                 mode='train',
                 name=model_name):
        """
        build basic dnn model
        @param info_dict: None,
        @param hyper_params: hyper parameters,
        @param reuse: reuse the variables or not
        @param is_saving: option for saving weights
        @param init_graph: initialize graph
        @param mode: enable a mode for run the model, 'train' or 'test'
        @param name: model name
        """
        super(BasicDNNModel, self).__init__()
        # model setup
        self.is_saving = is_saving
        self.init_graph = init_graph
        try:
            assert mode == 'train' or mode == 'test'
        except:
            raise AssertionError("'train' or 'test' mode, not others.")

        self.mode = mode
        if info_dict is not None:
            self.info_dict = info_dict
        else:
            self.info_dict = INFO
        self.info = utils.ParamWrapper(self.info_dict)
        if hyper_params is not None:
            self.hp_params_dict = hyper_params
        else:
            print('hyperparam is none')
            self.hp_params_dict = DNN_HP
        print('#################',self.hp_params_dict)
        self.hp_params = utils.ParamWrapper(self.hp_params_dict)
        self.lr = self.hp_params.learning_rate
        self.model_name = name

        if self.is_saving:
            self.save_dir = config.get('experiments', name.lower())
            self.save_dir_retrain = config.get('experiments', name.lower()) + 'retrain'

        # feature extraction
        self.feature_tp = self.info.feature_type # drebin
        self.feature_mp = self.info.feature_mapping_type # binary
        self.dataset_dir = self.info.dataset_dir

        self.mal_dir = os.path.join(self.dataset_dir, config.get('dataset', 'malware_dir_name'))
        self.ben_dir = os.path.join(self.dataset_dir, config.get('dataset', 'benware_dir_name'))
        if not (os.path.exists(config.get('feature.' + self.feature_tp, 'dataX')) and
                os.path.exists(config.get('feature.' + self.feature_tp, 'datay')) and
                os.path.exists(config.get('feature.' + self.feature_tp, 'vocabulary')) and
                os.path.exists(config.get('feature.' + self.feature_tp, 'normalizer')) and
                os.path.exists(config.get('dataset', 'name_list'))):
            self._data_preprocess()

        # obtain some hyper-parameters
        self.input_dim = len(
            utils.read_pickle(config.get('feature.' + self.feature_tp, 'vocabulary')))
        self.hidden_layers = self.hp_params.hidden_units
        self.output_dim = self.hp_params.output_dim
        tf.set_random_seed(self.hp_params.random_seed)
        if self.init_graph:
            self.model_graph(reuse=reuse)
    
    def get_parameter_size(self):
        """
        get the number of parameters
        """
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def _data_preprocess(self):
        """
        feature extraction
        """
        if (not os.path.exists(self.ben_dir)) and (not os.path.exists(self.mal_dir)):
            logger.error("directory '{}' or '{}' has no APK data.".format(self.ben_dir, self.mal_dir))
            return
        try:
            label_dict = self.get_label_dict()

            data_root_dir = config.get("dataset", "dataset_root")
            feat_save_dir = os.path.join(data_root_dir, "apk_data")
            get_droid_feature(self.ben_dir, feat_save_dir, feature_type=self.feature_tp)
            get_droid_feature(self.mal_dir, feat_save_dir, feature_type=self.feature_tp)

            feature_mapping = FeatureMapping(feat_save_dir, feature_type=self.feature_tp)
            naive_features, name_list = feature_mapping.load_features()

            if len(naive_features) == 0:
                logger.error("No features extracted.")
                return

            # remove S6: used permissions, this type of features depend on feature 'S7' APIs and feature 'S2' permission
            if not self.info.use_interdependent_features:
                naive_features = feature_mapping.remove_interdependent_featrues(naive_features)

            gt_label = np.array([label_dict[os.path.splitext(name.strip())[0]] \
                                 for name in name_list])

            vocab, vocab_info_dict, features = feature_mapping.generate_vocab(naive_features)

            # feature splitting as training dataset, validation dataset, testing dataset
            train_features, test_features, train_y, test_y, train_name_list, test_name_list = \
                train_test_split(features, gt_label, name_list, test_size=0.2, random_state=0)
            train_features, val_features, train_y, val_y, train_name_list, val_name_list = \
                train_test_split(train_features, train_y, train_name_list, test_size=0.25, random_state=0)

            # select frequent features
            vocab_selected, vocab_info_dict_selcted = \
                feature_mapping.select_feature(train_features, train_y, vocab, vocab_info_dict, dim=10000)
            MSG = "After feature selection, the feature number is {} vs. {}".format(len(vocab_selected), len(vocab))
            logger.info(msg=MSG)

            if self.feature_mp == 'count':
                training_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, train_features, status='train')
                val_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, val_features)
                test_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, test_features)
            elif self.feature_mp == 'binary':
                training_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, train_features, status='train')
                val_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, val_features)
                test_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, test_features)
            else:
                raise ValueError("Not supported")

            # save features and feature representations
            utils.dump_pickle(vocab_selected, config.get('feature.' + self.feature_tp, 'vocabulary'))
            utils.dump_pickle(vocab_info_dict_selcted, config.get('feature.' + self.feature_tp, 'vocab_info'))
            utils.dump_joblib([training_feature_vectors, val_feature_vectors, test_feature_vectors],
                              config.get('feature.' + self.feature_tp, 'dataX'))
            utils.dump_joblib([train_y, val_y, test_y],
                              config.get('feature.' + self.feature_tp, 'datay'))

            utils.write_whole_file('\n'.join(train_name_list + val_name_list + test_name_list),
                                   config.get('dataset', 'name_list'))
        except Exception as ex:
            logger.error(str(ex))
            sys.exit(1)

    def feature_extraction(self, apk_paths, is_ordering=True):
        """
        feature extraction
        @param apk_paths: the list of applications
        @param is_ordering: return the list of features corresponds to the apk_paths
        """
        feature_save_dir = os.path.join("/tmp", "apk_data")

        if os.path.exists(feature_save_dir):
            # delete the files related to features
            shutil.rmtree(feature_save_dir, ignore_errors=True)
            # a loosely checking
            # file_number = len(os.listdir(feature_save_dir))
            # assert file_number == len(apk_paths), "Feature extraction halts: there are feature files in directory '{}', and please remove it if it is not necessary anymore".format(feature_save_dir)

        get_droid_feature(apk_paths, feature_save_dir, feature_type=self.feature_tp)
        feature_mapping = FeatureMapping(feature_save_dir, feature_type=self.feature_tp)
        if is_ordering:

            feature = feature_mapping.preprocess_feature(is_ordering, apk_paths)

        else:

            feature = feature_mapping.preprocess_feature()

        if not os.path.exists(config.get('feature.' + self.feature_tp, 'vocabulary')):
            logger.warning("No vocabulary.")
            return np.array([])
        vocab = utils.read_pickle(config.get('feature.' + self.feature_tp, 'vocabulary'))

        if self.feature_mp == 'count':
            return feature_mapping.count_feature_mapping_normalized(vocab, feature)
        else:
            return feature_mapping.binary_feature_mapping_normalized(vocab, feature)

    def model_graph(self, reuse=False):
        """
        build the graph
        """
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='X')
        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None, ], name='Y')
        self.is_training = tf.placeholder(tf.bool, name="TRAIN")

        tf.set_random_seed(self.hp_params.random_seed)
        self.logits, self.y_tensor, self.dense1, self.dense2 = self.forward(self.x_input, self.y_input, reuse=reuse)
        self.model_inference()

    def forward(self, x_tensor, y_tensor, reuse=False):
        """
        let data pass through the neural network
        :param x_tensor: input data
        :type: Tensor.float32
        :param y_tensor: label
        :type: Tensor.int64
        :param reuse: Boolean
        :return: Null
        """
        self.nn = graph
        dense_1, dense_2, logits = graph(
            x_tensor, self.hidden_layers, self.output_dim,
            self.is_training, name=self.model_name, reuse=reuse
        )
        return logits, y_tensor, dense_1, dense_2

    def model_inference(self):
        """
        model inference
        """
        # loss definition
        self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_tensor,
            logits=self.logits
        )

        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_tensor,
            logits=self.logits
        )

        # prediction
        self.y_proba = tf.nn.softmax(self.logits)
        self.y_pred = tf.argmax(self.logits, axis=1)

        # some information
        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(self.y_pred, self.y_tensor))
        )

    def get_logits(self, dataX):
        """
        return logits for given data
        @param dataX 2D tensor
        """
        return self.nn(dataX,
                       self.hidden_layers,
                       self.output_dim,
                       is_training=False,
                       name=self.model_name,
                       reuse=True)[-1]

    def get_label_dict(self):
        """
        return label information, '1' means malicious examples and '0' means benign
        """
        def name_label_mapping(apk_dir, malicious=True):
            name_label_dict = {}
            for apk_path in list(utils.retrive_files_set(apk_dir, "", '|.apk')):
                apk_name = os.path.splitext(os.path.basename(apk_path))[0]
                if malicious:
                    name_label_dict[apk_name] = 1  # malicious label
                else:
                    name_label_dict[apk_name] = 0  # benign label
            return name_label_dict

        malware_label_mapping = name_label_mapping(self.mal_dir, malicious=True)
        benware_label_mapping = name_label_mapping(self.ben_dir, malicious=False)
        return dict(malware_label_mapping, **benware_label_mapping)

    def train(self, trainX=None, trainy=None, valX=None, valy=None):
        """train dnn"""
        if trainX is None or trainy is None or valX is None or valy is None:
            # load training dataset and validation dataset
            trainX, valX, _ = utils.read_joblib(config.get('feature.' + self.feature_tp, 'dataX')) #trainX.shape=(28683, 10000), valX.shape=(9562, 10000) ,All is 0 or 1 binaries
            trainy, valy, _ = utils.read_joblib(config.get('feature.' + self.feature_tp, 'datay')) #trainy.shape=(28683,), valy.shape=(9562,), 0 benign, 1 malicious.

        train_input = utils.DataProducer(trainX, trainy, self.hp_params.batch_size, n_epochs=self.hp_params.n_epochs)
        val_input = utils.DataProducer(valX, valy, self.hp_params.batch_size, name='val')

        global_train_step = tf.train.get_or_create_global_step()
        saver = tf.train.Saver()
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()

        # optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,
                                                                                      global_step=global_train_step)
        tf_cfg = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        tf_cfg.gpu_options.allow_growth = True
        tf_cfg.gpu_options.per_process_gpu_memory_fraction = 1.
        sess = tf.Session(config=tf_cfg)

        with sess.as_default():
            summary_writer = tf.summary.FileWriter(self.save_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            training_time = 0.0
            train_input.reset_cursor()
            output_steps = 50
            best_f1_val = 0.
            for step_idx, X_batch, y_batch in train_input.next_batch():
                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx + 1) % output_steps == 0:
                    print('Step {}/{}:{}'.format(step_idx + 1, train_input.steps, datetime.now()))
                    val_input.reset_cursor()
                    val_res_list = [sess.run([self.accuracy, self.y_pred], feed_dict={self.x_input: valX_batch,
                                                                                      self.y_input: valy_batch,
                                                                                      self.is_training: False}) \
                                    for [_, valX_batch, valy_batch] in val_input.next_batch()
                                    ]
                    val_res = np.array(val_res_list, dtype=object)
                    _acc = np.mean(val_res[:, 0])
                    _pred_y = np.concatenate(val_res[:, 1])
                    from sklearn.metrics import f1_score
                    _f1_score = f1_score(valy, _pred_y[:valy.shape[0]])

                    print('    validation accuracy {:.5}%'.format(_acc * 100))
                    print('    validation f1 score {:.5}%'.format(_f1_score * 100))

                    if step_idx != 0:
                        print('    {} samples per second'.format(
                            output_steps * self.hp_params.batch_size / training_time))
                        training_time = 0.

                    summary = sess.run(merged_summaries, feed_dict=train_dict)
                    summary_writer.add_summary(summary, global_train_step.eval(sess))

                    if best_f1_val < _f1_score:
                        best_f1_val = _f1_score
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        saver.save(sess,
                                   os.path.join(self.save_dir, 'checkpoint'),
                                   global_step=global_train_step)

                start = default_timer()
                sess.run(optimizer, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()

    def test_rpst(self, testX=None, testy=None, is_single_class=False):
        self.mode = 'test'
        if testX is None and testy is None:
            _, _, testX = utils.read_joblib(config.get('feature.' + self.feature_tp, 'dataX'))
            _, _, testy = utils.read_joblib(config.get('feature.' + self.feature_tp, 'datay'))

        if len(testX) == 0:
            print("No test data.")
            return

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        print('****************In test_rpst. Loading saved parameters from', self.save_dir)
        if cur_checkpoint is None:
            print("No saved parameters")
            return
        # load parameters
        saver = tf.train.Saver()
        eval_dir = os.path.join(self.save_dir, 'eval')
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            y_pred = tester(sess, testX, testy, self)
            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            accuracy = accuracy_score(testy, y_pred)
            b_accuracy = balanced_accuracy_score(testy, y_pred)

            MSG = "The accuracy on the test dataset is {:.5f}%"
            print(MSG.format(accuracy * 100))
            MSG = "The balanced accuracy on the test dataset is {:.5f}%"
            print(MSG.format(b_accuracy * 100))

            if not is_single_class:
                tn, fp, fn, tp = confusion_matrix(testy, y_pred).ravel()

                fpr = fp / float(tn + fp)
                fnr = fn / float(tp + fn)
                f1 = f1_score(testy, y_pred, average='binary')
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                    tf.Summary.Value(tag='f1 score', simple_value=f1)
                ])

                if not os.path.exists(eval_dir):
                    os.mkdir(eval_dir)
                summary_writer = tf.summary.FileWriter(eval_dir)
                summary_writer.add_summary(summary)

                print("Other evaluation metrics we may need:")
                MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
                print(MSG.format(fnr * 100, fpr * 100, f1 * 100))
            sess.close()
        return accuracy
    
    def test_pred_feat(self, testX=None, testy=None, is_single_class=False):
        self.mode = 'test'
        if testX is None and testy is None:
            _, _, testX = utils.read_joblib(config.get('feature.' + self.feature_tp, 'dataX'))
            _, _, testy = utils.read_joblib(config.get('feature.' + self.feature_tp, 'datay'))

        if len(testX) == 0:
            print("No test data.")
            return

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        print('****************In test_rpst. Loading saved parameters from', self.save_dir)
        if cur_checkpoint is None:
            print("No saved parameters")
            return
        # load parameters
        saver = tf.train.Saver()
        eval_dir = os.path.join(self.save_dir, 'eval')
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            y_pred, X_feat = tester(sess, testX, testy, self, feature=True)
            sess.close()
        return y_pred, X_feat

    def test(self, apks, gt_labels, is_single_class=False, is_save_feat=True):
        try:
            print('Extracting Features during test')
            testX = self.feature_extraction(apks)
            assert len(testX) == len(gt_labels), 'inconsistent first dimension'
            if is_save_feat:
                save_path = config.get('attack', 'advX')

                utils.dumpdata_np(testX, save_path)
        except Exception as ex:
            logger.error(str(ex))
            print("Feature extraction failed.")
            return 1

        return self.test_rpst(testX, gt_labels, is_single_class)

    def retrain(self, candidateX=None, candidatey=None, testX=None, testy=None, epochs=None):
        self.hp_params.n_epochs = epochs
        print('retraining for {0} epochs with batch size '.format(self.hp_params.n_epochs, self.hp_params.batch_size))
        # use candidate set for retraining

        train_input = utils.DataProducer(candidateX, candidatey, self.hp_params.batch_size, n_epochs=self.hp_params.n_epochs)
        val_input = utils.DataProducer(testX, testy, self.hp_params.batch_size, name='val')

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        print('***************In retrain function. Loading saved parameters from', self.save_dir)

        # load parameters
        saver = tf.train.Saver()
        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(self.save_dir_retrain, sess.graph)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()

        # optimizer
        global_train_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,global_step=global_train_step)

        with sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, cur_checkpoint)

            training_time = 0.0
            train_input.reset_cursor()
            output_steps = 1
            best_f1_val, best_acc_val = 0., 0.
            for step_idx, X_batch, y_batch in train_input.next_batch():
                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx) % output_steps == 0:
                    print('Step {}/{}:{}'.format(step_idx + 1, train_input.steps, datetime.now()))
                    val_input.reset_cursor()
                    val_res_list = [sess.run([self.accuracy, self.y_pred], feed_dict={self.x_input: valX_batch,
                                                                                      self.y_input: valy_batch,
                                                                                      self.is_training: False}) \
                                    for [_, valX_batch, valy_batch] in val_input.next_batch()
                                    ]
                    val_res = np.array(val_res_list, dtype=object)
                    _acc = np.mean(val_res[:, 0])
                    _pred_y = np.concatenate(val_res[:, 1])
                    from sklearn.metrics import f1_score

                    _f1_score = f1_score(testy, _pred_y[:testy.shape[0]])

                    #print('    validation accuracy {:.5}%'.format(_acc * 100))
                    #print('    validation f1 score {:.5}%'.format(_f1_score * 100))

                    if step_idx != 0:
                        #print('    {} samples per second'.format(
                        #    output_steps * self.hp_params.batch_size / training_time))
                        training_time = 0.

                    summary = sess.run(merged_summaries, feed_dict=train_dict)
                    summary_writer.add_summary(summary, global_train_step.eval(sess))
                    if best_acc_val < _acc:
                        best_acc_val = _acc

                    if best_f1_val < _f1_score:
                        best_f1_val = _f1_score

                        if not os.path.exists(self.save_dir_retrain):
                            os.makedirs(self.save_dir_retrain)
                        #saver.save(sess,
                        #           os.path.join(self.save_dir_retrain, 'checkpoint'),
                        #           global_step=global_train_step)
                start = default_timer()

                sess.run(optimizer, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()
        return best_acc_val

    def selection(self, budget=None, trainX=None, trainy=None, candidateX=None, candidateX_id=None, candidatey=None, candidatey_id=None, hybrid_test=None, hybrid_testy=None, metric=None, id_ratio=None, our_method_params=None):
        # select from hybrid candidate set
        self.mode = 'test'
        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        print('****************In test_rpst. Loading saved parameters from', self.save_dir)
        # load parameters
        saver = tf.train.Saver()
        sess = tf.Session()
        selected_candidateX, selected_candidatey = None, None
        with sess:
            saver.restore(sess, cur_checkpoint)
            if metric == 'random':
                selected_candidateX, selected_candidatey = Random(sess, candidateX, candidatey, budget)
            elif metric == 'deepgini':
                selected_candidateX, selected_candidatey, _ = deepgini(sess, candidateX, candidatey, self, budget)
            elif metric == 'entropy':
                selected_candidateX, selected_candidatey, _ = entropy(sess, candidateX, candidatey, self, budget)
            elif metric == 'gd':
                selection_size = int(budget * candidateX.shape[0])
                selected_candidateX, selected_candidatey = GD(sess, candidateX, candidatey, self, selection_size, 60)
            elif metric == 'dat':
                selected_candidateX, selected_candidatey = dat(sess, candidateX, candidatey, candidateX_id, candidatey_id, hybrid_test, hybrid_testy, self, id_ratio, budget)
            elif metric == 'ensemble':
                selected_candidateX, selected_candidatey = ensemble(sess, trainX, trainy, candidateX, candidatey, candidateX_id, candidatey_id, hybrid_test, hybrid_testy, self,
                         id_ratio, budget)
            elif metric == 'ensemble_simple':
                selected_candidateX, selected_candidatey = ensemble_simple(sess, candidateX, candidatey, self, budget)
            elif metric == 'ensemble_simple_add':
                selected_candidateX, selected_candidatey = ensemble_simple_add(sess, candidateX, candidatey, self, budget)
            elif metric == 'ensemble_ood_detector':
                selected_candidateX, selected_candidatey = ensemble_ood_detector(sess, trainX, trainy, candidateX,
                                                                                 candidatey,
                                                                                 hybrid_test,
                                                                                 hybrid_testy, self,
                                                                                 budget)
            elif metric == 'dat_ood_detector':
                selected_candidateX, selected_candidatey = dat_ood_detector(sess, trainX, trainy, candidateX,
                                                                            candidatey,
                                                                            hybrid_test,
                                                                            hybrid_testy, self,
                                                                            budget)
            elif metric == 'ensemble_ood_gd':
                selected_candidateX, selected_candidatey = ensemble_ood_gd(sess, trainX, trainy, candidateX, candidatey, self, budget)
            elif metric == 'ensemble_algorithm_3':
                selected_candidateX, selected_candidatey = ensemble_algorithm_3(sess, trainX, trainy, candidateX,
                                                                                candidatey, self, budget)
            elif metric == 'ensemble_algorithm_3_batch':
                selected_candidateX, selected_candidatey = ensemble_algorithm_3_batch(sess, candidateX, candidatey,
                                                                                      self, budget, our_method_params)
            elif metric == 'ensemble_p_values_incre_change':
                selected_candidateX, selected_candidatey = ensemble_p_values_incre_change(sess, trainX, trainy, candidateX, candidatey, self, budget, our_method_params)
            elif metric == 'ensemble_p_values':
                selected_candidateX, selected_candidatey = ensemble_p_values(sess, trainX, trainy, candidateX,
                                                                             candidatey, self, budget,
                                                                             our_method_params)
            elif metric == 'ensemble_ood_uncertain':
                selected_candidateX, selected_candidatey = ensemble_ood_uncertain(sess, trainX, trainy, candidateX, candidatey, hybrid_test, hybrid_testy, self, budget)
            sess.close()

        return selected_candidateX, selected_candidatey


def _main():
    basic_dnn = BasicDNNModel()
    basic_dnn.train()
    basic_dnn.test_rpst()


if __name__ == "__main__":
    _main()
