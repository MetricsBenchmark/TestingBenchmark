# TestingBenchmark
This repository contains experiments conducted in the paper ''.

## Conceptual Visualization of Clusters

Conceptual visualization of clusters:  

- Each **red ball** represents a *mispredicted* sample.  
- Each **green ball** represents a *correctly predicted* sample.  
- A **black circle** encompassing red balls denotes a *cluster*.

![Cluster Visualization](Concept_cluster.png)


## Hyperparameters of the Three-Step Clustering Pipeline

### Global Search Space & Selections
![Cluster Visualization](Parameters.png)

### Dataset-Specific Results

- **AndroZoo:** 
  - `HAC: K = 10`, `K-Means: K = 9`.
- **AndroZoo & Drebin:** 
  - `HAC: K = 10`, `K-Means: K = 10`.
- **IMDb:**  
  - UMAP: `#Components = 50`, `Min. dist. = 0.1`, `#Neighbors = 15`  
  - PCA: `#Components = 50`  
  - HDBSCAN: `Min. cluster size = 2`  
  - HAC (knee-point): `#Clusters = 13`  
  - K-Means (knee-point): `#Clusters = 13`
- **MNIST:**  
  - UMAP: `#Components = 25`, `Min. dist. = 0.1`, `#Neighbors = 15`
- **Udacity:**  
  - Dimensionality reduction (PCA/GRP): `#Components = 50`  
  - HAC: `K = 17`  
  - K-Means: `K = 13`

### Abbreviations

- **UMAP** — Uniform Manifold Approximation and Projection  
- **PCA** — Principal Component Analysis  
- **GRP** — Gaussian Random Projection  
- **DBSCAN** — Density-Based Spatial Clustering of Applications with Noise  
- **HDBSCAN** — Hierarchical DBSCAN  
- **HAC** — Hierarchical Agglomerative Clustering 


### Parameter Glossary
- **`#` = “Number of …”** — legacy shorthand that appears in some figures/tables (e.g., `#Components`, `#Neighbors`).
- **K** — number of clusters (used by HAC / K-Means).  
- **#Components** — projected dimensionality (UMAP / PCA / GRP).  
- **Min. dist.** — minimum distance between embedded points (UMAP).  
- **#Neighbors** — size of local neighborhood used for manifold approximation (UMAP). 
- **min_neighbors** — minimum number of neighbors within the distance cutoff for a point to be a core point (**DBSCAN**).
- **Dist. Threshold** — distance cutoff for neighborhood connectivity.
- **Min. cluster size** — minimum number of points required to form a cluster (HDBSCAN). 


### Selection Strategy Glossary
- **knee-point** — model selection at the elbow/knee of a metric curve [1].
- **grid search** — hyperparameter tuning method that exhaustively tries all candidate values within a predefined set [2].  



## Detailed implementation of test selection metrics

We group prior metrics into three families: **uncertainty-based**, **diversity-based**, and **surprise-based**.

### Uncertainty-based

- **DeepGini (Gini)** — selects most uncertain samples using output probabilities [3].  
  **Implementation.** Use model softmax over all classes (classification only).  
  **Hyperparams.** 

- **Entropy (Ent)** — Shannon-entropy of predictive distribution; higher entropy ⇒ higher uncertainty [4].  
  **Implementation.** Use model softmax over all classes (classification only).  
  **Hyperparams.** 

### Diversity-based

- **Neuron Coverage (NC)** — prioritizes inputs that activate more neurons above a threshold [5].  
  **Implementation.** Compute per-input incremental coverage across hidden layers; rank by coverage gain.  
  **Hyperparams.** Activation threshold `t = 0.75` [3, 6].

- **K-Multisection Neuron Coverage (KMNC)** — partitions each neuron’s value range into `K` bins; more bins hit ⇒ more diverse [7].  
  **Implementation.** To reduce cost, compute on the **last hidden layer** for all datasets.  
  **Hyperparams.** `K = 100`.

- **Geometric Diversity (GD)** — favors subsets with large determinant in an embedding space (diverse geometry) [8].  
  **Implementation.**  
  - *MNIST / Udacity*: extract features using **VGG-16**, take activations **after the last conv layer** (black-box).  
  - *AndroZoo*: use **DeepDrebin** features from the **first hidden layer**.  
  - Prefer shallower features to capture input-level diversity [9].  
  **Hyperparams.** 

- **Standard Deviation (STD)** — selects sets with larger per-feature variability [8].  
  **Implementation.** Compute L2-norm of per-feature standard deviations on the **same embeddings as GD**.  
  **Hyperparams.** 

### Surprise-based

- **Likelihood-based Surprise Adequacy (LSA)** — KDE density on activation traces; lower density ⇒ more “surprising” [10].  
  **Implementation.** Use **last hidden layer** activations; reference = **training set**; KDE for density.  
  **Hyperparams.** Kernel/bandwidth: default from the implementation (no manual tuning reported).

- **Distance-based Surprise Adequacy (DSA)** — distance in activation space to nearest training sample (nearest neighbor) [10].  
  **Implementation.** Use **last hidden layer** activations; classification only (requires class boundaries).  
  **Hyperparams.** 


### Performance Estimation Metrics

- **Cross Entropy-based Sampling (CES)** — selects a set whose distribution matches the whole test set by minimizing cross entropy [11].  
  **Implementation.** Optimize selection to approximate test distribution in the chosen embedding space; use the same **budget** as other methods.  
  **Hyperparams.** Sample budget (matches others).

- **Practical Accuracy Estimation (PACE)** — cluster tests then sample representatives adaptively [12].  
  **Implementation.** Cluster embeddings (UMAP → K-Means/HAC); choose **K via knee-point** and sample per-cluster adaptively.  
  **Hyperparams.** `K` chosen by knee-point (no fixed `K`).

- **DeepReduce (DR)** — multi-objective selection balancing adequacy and distribution similarity with minimal data [13].  
  **Implementation.** Use the authors’ objectives; match selection **budget** to baselines.  
  **Hyperparams.** Sample budget (matches others).

- **DeepEST (EST)** — adaptive sampling for accuracy estimation and fault detection [14].  
  **Implementation.**  
  - *Classification*: auxiliary variable = **DSA + confidence** (best trade-off reported).  
  - *Regression*: no softmax & DSA undefined ⇒ use **LSA** as auxiliary variable.  
  **Hyperparams.** None beyond the chosen auxiliary variable.


### Retraining-oriented Metrics

- **Multiple-Boundary Clustering & Prioritization (MCP)** — prioritize boundary-region samples via top-2 predicted classes; sample evenly across clusters [15].  
  **Implementation.** Partition by **top-2 softmax classes**; not applicable to **regression**.  
  **Hyperparams.** None.

- **Distribution-Aware Test Selection (DAT)** — hybrid: uncertain **in-distribution** + random **out-of-distribution** [16].  
  **Implementation (OOD detection).** Semi-supervised, distance-based OOD on **intermediate features** using pre-trained models [17, 18].  
  - *MNIST*: **LeNet-1**, **2nd hidden layer**; centroid distance threshold; **AUC-ROC 85.09%** with **Fashion-MNIST** as OOD (per Hu et al., 2022).  
  - *AndroZoo*: **DeepDrebin**, **2nd hidden layer**; centroid distance threshold; **AUC-ROC 99.78%** with **FGSM** adversarial OOD.  
  - *IMDb*: raw/text features separate ID vs. corrupted OOD; **AUC-ROC 70.89%**.  
  **Scope.** **Classification only**.  
  **Hyperparams.** OOD distance threshold chosen on validation (per dataset).


### Baseline

- **Random Selection (Rand)** — sample uniformly at random without replacement.  
  **Implementation.** Uniform random from test set.



## References
[1]: Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). *Finding a “Kneedle” in a Haystack: Detecting Knee Points in System Behavior*. ICDCS Workshops, 166–171.
[2]: Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. JMLR, 13, 281–305.  
[3]: Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. In KDD, Vol. 96, 226–231.  
[4]: Guerriero, A., Pietrantuono, R., & Russo, S. (2024). *DeepSample: DNN sampling-based testing for operational accuracy assessment*. In Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, 1–12.  
[5]: Aghababaeyan, Z., Abdellatif, M., Dadkhah, M., & Briand, L. (2024). *DeepGD: A multi-objective black-box test selection approach for deep neural networks*. ACM Transactions on Software Engineering and Methodology, 33(6), 1–29.  
[6]: Anglin, K., Liu, Q., & Wong, V. C. (2024). *A primer on the validity typology and threats to validity in education research*. Asia Pacific Education Review, 25(3), 557–574.  
[7]: Gundling, C. (2016). *Steering angle model: Cg32*. https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/cg23  
[8]: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding*. In NAACL-HLT (Vol. 1: Long and Short Papers), 4171–4186.  
[9]: Liang, J., Elbaum, S., & Rothermel, G. (2018). *Redefining prioritization: Continuous prioritization for continuous integration*. In Proceedings of the 40th International Conference on Software Engineering, 688–698.  
[10]: Gao, X., Feng, Y., Yin, Y., Liu, Z., Chen, Z., & Xu, B. (2022). *Adaptive test selection for deep neural networks*. In Proceedings of the 44th International Conference on Software Engineering, 73–85.  
[11]: Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv preprint arXiv:1412.6572.  
[12]: Gildenblat, J. (2016). *Visualizations for understanding the regressed wheel steering angle for self-driving cars*. https://github.com/jacobgil/keras-steering-angle-visualizations  
[13]: Guerriero, A., Pietrantuono, R., & Russo, S. (2021). *Operation is the hardest teacher: Estimating DNN accuracy looking for mispredictions*. In Proceedings of the IEEE/ACM 43rd International Conference on Software Engineering (ICSE).  
[14]: Attaoui, M., Fahmy, H., Pastore, F., & Briand, L. (2023). *Black-box safety analysis and retraining of DNNs based on feature extraction and clustering*. ACM Transactions on Software Engineering and Methodology, 32(3), 1–40.  
[15]: Allix, K., Bissyandé, T. F., Klein, J., & Le Traon, Y. (2016). *AndroZoo: Collecting millions of Android apps for the research community*. In Proceedings of the 13th International Conference on Mining Software Repositories.  
[16]: Allix, K., Bissyandé, T. F., Klein, J., & Le Traon, Y. (2016). *AndroZoo online repository*. https://androzoo.uni.lu/  
[17]: Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A robustly optimized BERT pretraining approach*. arXiv preprint arXiv:1907.11692.  
[18]: Ma, L., Juefei-Xu, F., Zhang, F., Sun, J., Xue, M., Li, B., Chen, C., Su, T., Li, L., Liu, Y., et al. (2018). *DeepGauge: Multi-granularity testing criteria for deep learning systems*. In Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering, 120–131.  
