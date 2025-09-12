from learner import model_scope_dict
from tools import utils
import numpy as np
import os
import csv
from config import config

model_name = 'deepdrebin'
retrain_type = 'type1'
test_set = 'hybrid'


class datastore:
    def __init__(self, X_path, y_path, file_type):
        self.X_path = X_path
        self.y_path = y_path
        self.file_type = file_type
        self.__load()

    def __load(self):
        if self.file_type == 'pkl':
            self.trainX, self.valX, self.testX = utils.read_joblib(self.X_path)
            self.trainy, self.valy, self.testy = utils.read_joblib(self.y_path)
            self.X, self.y = self.testX, self.testy
        else:
            self.X = utils.readdata_np(self.X_path)
            self.y = utils.readdata_np(self.y_path)


set_id = datastore(  # 64007 + 21336 + 21336
    './androzoo/drebin/X.pkl',
    '/androzoo/drebin/y.pkl',
    'pkl'
)

set_ood = {
    'drebin': datastore(  # 28683 + 9562 + 9562
        './drebin/drebin/X.pkl',
        './drebin/drebin/y.pkl',
        'pkl'
    ),
    'ad2018': datastore(  # 8367
        './androzoo/2018/feature/X.data',
        './androzoo/2018/feature/y.data',
        'data'
    ),
    'ad2019': datastore(  # 8076
        './androzoo/2019/feature/X.data',
        './androzoo/2019/feature/y.data',
        'data'
    )
}

TYPE1 = 'type1'
TYPE2 = 'type2'
HYBRID = 'hybrid'
ORIGINAL = 'original'
DEEPDREBIN = 'deepdrebin'
BASIC_DNN = 'basic_dnn'

targeted_model_names_dict = model_scope_dict.copy()
targeted_model_name = model_name
targeted_model = targeted_model_names_dict[targeted_model_name](mode='test')

project_root = config.get('DEFAULT', 'project_root')
dataset_root = config.get('dataset', 'dataset_root')
log_path = os.path.join(project_root, 'results')
log_file = os.path.join(log_path, 'uncertainty.csv')

feat_path = dataset_root
if model_name == DEEPDREBIN:
    feat_path = os.path.join(feat_path, 'attack', 'adversarial_samples')
else:
    feat_path = os.path.join(feat_path, 'attack_b_dnn', 'adversarial_samples')

# Note that pristine feats are the same under each attack methods.
pristine_feat_path = os.path.join(feat_path, 'fgsm', 'pristine_l-infinity.data')
pristine_feat = utils.readdata_np(pristine_feat_path)
# pristine_feat = pristine_feat[:int(_pristine_feat.shape[0]*sample_portion)]
test_suite = config.get('DEFAULT', 'test_suite')
test_suite_dir = test_suite

methods = ['fgsm', 'PGD-linf', 'PGD-l2', 'PGD-l1', 'PGD-Adam', 'GDKDE', 'BCA_K', 'BGA_K', 'GROSSE', 'JSMA', 'MAX',
           'MIMICRY', 'POINTWISE', 'fgsml1', 'fgsml2']
perturbed_paths = [
    os.path.join(feat_path, 'fgsm', 'fgsm_l-infinity.data'),
    os.path.join(feat_path, 'pgdlinf', 'pgdlinf_l-infinity.data'),
    os.path.join(feat_path, 'pgdl2', 'pgdl2_l2.data'),
    os.path.join(feat_path, 'pgdl1', 'pgdl1_.data'),
    os.path.join(feat_path, 'pgd_adam', 'pgd_adam_.data'),
    os.path.join(feat_path, 'gdkde', 'gdkde_.data'),
    os.path.join(feat_path, 'bca_k', 'bca_k_.data'),
    os.path.join(feat_path, 'bga_k', 'bga_k_.data'),
    os.path.join(feat_path, 'grosse', 'grosse_.data'),
    os.path.join(feat_path, 'jsma', 'jsma_.data'),
    os.path.join(feat_path, 'max', 'max_.data'),
    os.path.join(feat_path, 'mimicry', 'mimicry_.data'),
    os.path.join(feat_path, 'pointwise', 'pointwise_.data'),
    os.path.join(feat_path, 'fgsml1', 'fgsml1_l1.data'),
    # os.path.join(feat_path, 'fgsml2', 'fgsml2_l2.data')
]

SIZE = 8000
HALF_TEST_SIZE = 21336 // 2


def make_ood_test(ood_name):
    if ood_name == 'drebin':
        np.random.seed(4269)
    elif ood_name == 'ad2018':
        np.random.seed(2018)
    elif ood_name == 'ad2019':
        np.random.seed(2019)

    if ood_name == 'drebin':
        selected_id = np.random.choice(set_id.X.shape[0], HALF_TEST_SIZE, replace=False)
        selected_ood = np.random.choice(set_ood[ood_name].trainX.shape[0], HALF_TEST_SIZE, replace=False)
        idX, idy = set_id.X[selected_id], set_id.y[selected_id]
        oodX, oody = set_ood[ood_name].trainX[selected_ood], set_ood[ood_name].trainy[selected_ood]
    else:
        selected_id = np.random.choice(set_id.X.shape[0], SIZE, replace=False)
        selected_ood = np.random.choice(set_ood[ood_name].X.shape[0], SIZE, replace=False)
        idX, idy = set_id.X[selected_id], set_id.y[selected_id]
        oodX, oody = set_ood[ood_name].X[selected_ood], set_ood[ood_name].y[selected_ood]
    X, y = np.concatenate((idX, oodX), axis=0), np.concatenate((idy, oody), axis=0)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    # assert X.shape[0] == SIZE * 2
    return X, y


def make_pertubed_test():
    oodX, oody = [None] * 2
    for i, perturbed_path in enumerate(perturbed_paths):
        if perturbed_path is None:
            continue
        if not os.path.exists(perturbed_path):
            print('Perturbed path does not exist: %s' % perturbed_path)
        perturbed_feat = utils.readdata_np(perturbed_path)
        if i == 0:
            oodX, oody = perturbed_feat, set_id.y
        else:
            oodX = np.concatenate((oodX, perturbed_feat), axis=0)
            oody = np.concatenate((oody, set_id.y), axis=0)
    selected_id = np.random.choice(set_id.X.shape[0], HALF_TEST_SIZE, replace=False)
    selected_ood = np.random.choice(oodX.shape[0], HALF_TEST_SIZE, replace=False)
    idX, idy = set_id.X[selected_id], set_id.y[selected_id]
    oodX, oody = oodX[selected_ood], oody[selected_ood]
    X, y = np.concatenate((idX, oodX), axis=0), np.concatenate((idy, oody), axis=0)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    assert X.shape[0] == HALF_TEST_SIZE * 2
    return X, y


def make_label_shift_test():
    X, y = set_id.X, set_id.y
    malware_idx = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0]
    goal = HALF_TEST_SIZE * 2
    malware_goal = int(goal * 0.8)
    # randomly duplicate malware samples until we reach the goal
    integer_times = malware_goal // malware_idx.shape[0]
    malware_idx = np.tile(malware_idx, integer_times)
    malware_idx = np.concatenate(
        (malware_idx, np.random.choice(malware_idx, malware_goal % malware_idx.shape[0], replace=False)), axis=0)
    benign_goal = goal - malware_goal
    benign_idx = np.random.choice(benign_idx, benign_goal, replace=False)
    assert malware_idx.shape[0] + benign_idx.shape[0] == goal
    X = np.concatenate((X[malware_idx], X[benign_idx]), axis=0)
    y = np.concatenate((y[malware_idx], y[benign_idx]), axis=0)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    assert X.shape[0] == goal
    return X, y


def test(test_set, X, y):
    acc = targeted_model.test_rpst(testX=X, testy=y, is_single_class=True)
    with open(log_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([model_name, test_set, acc, X.shape[0]])

def main():
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name', 'test_set', 'accuracy', 'test_size'])

    # test('androzoo', set_id.X, set_id.y)
    for ood_name in set_ood.keys():
        test(ood_name, *make_ood_test(ood_name))

    # test('drebin', *make_ood_test('drebin'))
    # test('adversarial', *make_pertubed_test())
    # test('label_shift', *make_label_shift_test())

    # print('ID test set 0/1 distribution: %d/%d' % (np.sum(set_id.y == 0), np.sum(set_id.y == 1)))
    # print('Drebin test set 0/1 distribution: %d/%d' % (np.sum(set_ood['drebin'].y == 0), np.sum(set_ood['drebin'].y == 1)))


if __name__ == '__main__':
    main()
