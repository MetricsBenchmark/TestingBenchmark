'''
This code examines each candidate pipeline for IMDb dataset
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig, ElectraModel
from tqdm import tqdm  # for progress bar
from DBCV.DBCV import DBCV
import cuml
from cuml import DBSCAN
from torchvision import models
import sklearn
import cluster
from draw_heatmap import heatmap_sorted_by_cluster_similarity
from cuml import PCA
from cuml.random_projection import GaussianRandomProjection
from cuml.cluster import KMeans
from cuml import AgglomerativeClustering

def decode_texts():
    tr = load_dataset("imdb", split="train")
    nominal_train = tr["text"]
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(nominal_train)
    decoded_texts = tokenizer.sequences_to_texts(np.load('data/tokenized/testX.npy'))
    np.save(f'data/tokenized/testX_decoded.txt', decoded_texts)

def pretrained_bert(): # 'tokenized is original data
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    decoded_advX = np.load('./data/tokenized/testX_decoded.txt.npy')
    labels = np.load('./data/tokenized/testy.npy')

    tokens_train = tokenizer.batch_encode_plus(decoded_advX.tolist(),max_length=25,pad_to_max_length=True,truncation=True)
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(labels.tolist())
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=128)
    feats = []
    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False
    for step, batch in enumerate(train_dataloader):
        sent_id, mask, labels = batch
        # get model predictions for the current batch
        preds = bert(sent_id, mask)
        feat = preds['last_hidden_state'].reshape((len(preds['last_hidden_state']), -1))
        feats.append(feat.cpu().numpy())
    feature_np = np.concatenate(np.array(feats), axis=0)
    np.save(f'./data/tokenized/feat_bert.npy', feature_np)

def pretrained_roberta_fast(decoded_data, batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    config = AutoConfig.from_pretrained("FacebookAI/roberta-base")
    config.is_decoder = True
    model = RobertaForCausalLM.from_pretrained("FacebookAI/roberta-base", config=config)
    model.eval()
    target_layer = 11
    decoded_advX = decoded_data
    all_embeddings = []
    for i in tqdm(range(0, len(decoded_advX), batch_size)):
        batch_texts = decoded_advX[i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=25)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            layer_embeddings = outputs.hidden_states[target_layer]
        pooled = layer_embeddings.mean(dim=1).cpu().numpy()
        all_embeddings.append(pooled)
    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

def pretrained_roberta_fast_save(shift='tokenized', batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    config = AutoConfig.from_pretrained("FacebookAI/roberta-base")
    config.is_decoder = True
    model = RobertaForCausalLM.from_pretrained("FacebookAI/roberta-base", config=config)
    model.eval()
    target_layer = 11
    decoded_advX = np.load('./data/tokenized/testX_decoded.txt.npy')
    all_embeddings = []

    for i in tqdm(range(0, len(decoded_advX), batch_size)):
        batch_texts = decoded_advX[i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=25)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            layer_embeddings = outputs.hidden_states[target_layer]
        pooled = layer_embeddings.mean(dim=1).cpu().numpy()
        all_embeddings.append(pooled)
    all_embeddings = np.vstack(all_embeddings)
    np.save(f'./data/tokenized/feat_roberta.npy', all_embeddings)
    return all_embeddings

def pretrained_electra_fast(shift='tokenized', batch_size=64):
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    model = ElectraModel.from_pretrained("google/electra-base-discriminator")
    model.eval()
    target_layer = 11
    decoded_advX = np.load('./data/tokenized/testX_decoded.txt.npy')
    all_embeddings = []

    for i in tqdm(range(0, len(decoded_advX), batch_size)):
        batch_texts = decoded_advX[i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=25)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            layer_embeddings = outputs.hidden_states[target_layer]

        pooled = layer_embeddings.mean(dim=1).cpu().numpy()
        all_embeddings.append(pooled)
    all_embeddings = np.vstack(all_embeddings)
    np.save(f'./data/tokenized/feat_electra.npy', all_embeddings)
    return all_embeddings, labels

def test_pipelines(data='tokenized', model_name='gru', FE = 'bert', DR='umap', CA='dbscan'):
    mispred_idx = np.load(f'./data/{data}/{model_name}_mispred_idx.npy')
    if FE == 'bert':
        feat = np.load(f'./data/{data}/feat_bert.npy')
    # 4. use pretrained Roberta for FE
    elif FE == 'roberta':
        feat = np.load(f'./data/{data}/feat_roberta.npy') #feat_roberta
    elif FE == 'electra':
        feat = np.load(f'./data/{data}/feat_electra.npy')
    suite_feat = feat[mispred_idx]
    suite_feat = suite_feat.reshape((len(suite_feat)), -1)
    if DR == 'UMAP':
        u = cluster.umap_gpu(ip_mat=suite_feat, min_dist=0.1, n_components=min(50, len(suite_feat) - 1), n_neighbors=15,
                             metric='Euclidean')
    elif DR == 'PCA':
        pca_float = PCA(n_components=50)
        u = pca_float.fit_transform(suite_feat)
    elif DR == 'GRP':
        GRP_float = GaussianRandomProjection(n_components=50, random_state=42)
        u = GRP_float.fit_transform(suite_feat)

    if CA == 'dbscan':
        optimal_eps = cluster.find_optimal_eps(u)
        dbscan_float = DBSCAN(eps=optimal_eps, min_samples=2)
        labels = dbscan_float.fit_predict(u)
    elif CA == 'hdbscan':
        hdbscan_float = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = hdbscan_float.fit_predict(u)
    elif CA == 'HAC':
        HAC = AgglomerativeClustering(n_clusters=13)
        labels = HAC.fit_predict(u)
    elif CA == 'Kmeans':
        kmeans_float = KMeans(n_clusters=13)
        labels = kmeans_float.fit_predict(u)
    heatmap_sorted_by_cluster_similarity(u, labels,
                                         f'./heatmap/save/heatmap_similarity_ordered_{DR}_{CA}_{FE}_{data}.png')

    silhouette_umap = sklearn.metrics.silhouette_score(u, labels)
    DBCV_score = DBCV(u, labels)
    clustering_results = {
        "Number of Clusters": labels.max() + 1,
        "Silhouette Score": silhouette_umap,
        "DBCV Score": DBCV_score,
        "Combined Score": 0.5 * silhouette_umap + 0.5 * DBCV_score,
        "Number of Mispredicted Inputs": len(u),
        "Number of Noisy Inputs": list(labels).count(-1)
    }
    print(clustering_results)
    print(model_name, FE)

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
    test_pipelines(args.model, args.FE, args.DR, args.CA)