"""
classifier implementation: data preprocessing and model learning
"""
from collections import defaultdict
from learner import feature_extractor

from learner.basic_DNN import BasicDNNModel, DNN_HP
from learner.b_DNN import B_DNNModel

_model_scope_dict = {
    'deepdrebin': BasicDNNModel,
    'basic_dnn': B_DNNModel,
}

model_scope_dict = defaultdict(**_model_scope_dict)


