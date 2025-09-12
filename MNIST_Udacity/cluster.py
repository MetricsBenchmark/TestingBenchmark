import sklearn
from PIL import Image
import numpy as np
from torchvision import transforms, models
import torch
import os
def extract_features_torch(images, dataset, device='cpu'):
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    model = FeatureExtractor(resnet50)  # already included AdaptiveAvgPool2d
    model.to(device)
    model.eval()
    image_features = []
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Matches ResNet input
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                             std=[0.229, 0.224, 0.225])
    ])
    for image in images:
        if 'mnist' in dataset:
            image = (image * 255).astype(np.uint8)
            image = np.repeat(image, 3, axis=2)
        elif 'udacity' in dataset and dataset != 'udacity_adv':
            image = Image.open(image).convert('RGB')  # 0~255 image

        img_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_tensor).cpu().numpy()
        image_features.append(features)
    image_features = np.squeeze(np.array(image_features, dtype=np.float64), axis=1)
    print('image_features.shape', image_features.shape)
    return image_features

from sklearn.preprocessing import StandardScaler
from cuml.manifold import UMAP
def umap_gpu(ip_mat, min_dist, n_components, n_neighbors, metric):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    scaler = StandardScaler()
    ip_std = scaler.fit_transform(ip_mat)
    reducer = UMAP(min_dist=min_dist, n_components=n_components, n_neighbors=n_neighbors, metric=metric)
    umap_embed = reducer.fit_transform(ip_std)
    return umap_embed


import torch.nn as nn
class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC layer

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x


from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
def find_optimal_eps(X):
    # Assume X is your dataset: shape (n_samples, n_features)
    # Use 2 nearest neighbors: the point itself and its closest neighbor
    # usage: find_optimal_eps(u)
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    # Get the distance to the closest neighbor (ignore self-distance at index 0)
    nearest_distances = np.sort(distances[:, 1])
    # Use KneeLocator to find the elbow point
    kneedle = KneeLocator(range(len(nearest_distances)), nearest_distances, S=1.0, curve='convex', direction='increasing')
    optimal_eps = nearest_distances[kneedle.knee] if kneedle.knee is not None else None
    if optimal_eps == None:  # use default value if None
        optimal_eps = 0.5
    return optimal_eps