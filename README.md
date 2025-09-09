# TestingBenchmark
This repository contains experiments conducted in the paper ''.

## Conceptual Visualization of Clusters

Conceptual visualization of clusters:  

- Each **red ball** represents a *mispredicted* sample.  
- Each **green ball** represents a *correctly predicted* sample.  
- A **black circle** encompassing red balls denotes a *cluster*.

![Cluster Visualization](Concept_cluster.png)


## Hyperparameters of the Three-Step Clustering Pipeline

### Dataset-Specific Results

| Methods   | Descriptions          | Candidate values    | Selection   | AndroZoo | IMDb | MNIST | Udacity |
|-----------|-----------------------|---------------------|-------------|----------|-------------------|----------------------|-----------------------|---------------------|
| **UMAP**  | Number of Components  | [5,10,25,50,75,100] | grid search | 25        | 50                   | 25                    | 50                   |
|           | Minimum Distance      | [0.01,0.1,0.3,0.5]  | grid search | 0.1        | 0.1                  | 0.1                   | 0.1                   |
|           | Number of Neighbors   | [5,15,25,50]        | grid search | 15        | 15                   | 15                    | 15                   |
| **PCA**   | Number of Components  | [5,10,25,50,75,100] | grid search | 25        | 50                   | –                     | –                  |
| **GRP**   | Number of Components  | [5,10,25,50,75,100] | grid search | 25        |  50                    | –                     |  –                |
| **DBSCAN**| Minimum Neighbors     | [2,5,10,15]         | grid search | 2        |  2                    | –                     | –                   |
|           | Distance Threshold    | -                   | knee-point  | Varied   | Varied               | Varied                 | Varied              |
| **HDBSCAN**| Minimum Cluster Size  | [2,5,10,15]         | grid search | 2       | 2                    | –                     | –                   |
| **HAC**   | Number of Clusters    | range[5,35]         | knee-point  | 10       |13                   | –                      | –                  |
| **K-Means**| Number of Clusters    | range[5,35]         | knee-point  | 9       | 13                   | –                     | –                  |


### Selection Strategy Description
- **knee-point** — a hyperparameter tuning method that selects the optimal value at the elbow/knee of a metric curve [1].
- **grid search** — a hyperparameter tuning method that exhaustively tries all candidate values within a predefined set [2].  

## Detailed implementation of test selection metrics

### Uncertainty-based
- **DeepGini (Gini, 2020)** — implemented using the softmax layer output as the class probabilities from each studied classifier.  
- **Entropy (Ent, 2014)** — implemented using the softmax layer output as the class probabilities from each studied classifier.  

### Diversity-based
- **Neuron Coverage (NC, 2017)** — For all datasets, we use neurons in the `last hidden layer` to reduce computational cost, and set the neuron activation threshold at `t = 0.75`.
- **K-Multisection Neuron Coverage (KMNC, 2018)** — We use the `last hidden layer` for all datasets to reduce computational cost, and we divide coverage for each neuron into `K = 100` sections.
- **Geometric Diversity (GD, 2023)**
  - *MNIST / Udacity*: We follow the original implementation, which uses **VGG-16** to extract features for image data in a black-box manner. Specifically, we use the activation value on the **layer after the last convolutional layer** as features.  
  - *AndroZoo*: we use pre-trained **DeepDrebin** model with the **first hidden layer** to extract features. We use shallow-layer features to capture input-level diversity [9].
  -  *IMDb*: we use pre-trained **Transformer** model with the **first hidden layer** to extract features. We use shallow-layer features to capture input-level diversity [9].
- **Standard Deviation (STD)** — The feature extraction is the same as GD.  

### Surprise-based
- **Likelihood-based Surprise Adequacy (LSA)** — For MNIST, AndroZoo, and IMDb, we use **last hidden layer** activations; For Udacity, we use the **penultimate hidden layer** activations.
- **Distance-based Surprise Adequacy (DSA)** — For all classification datasets (MNIST, AndroZoo, IMDb), we use **last hidden layer** activations; DSA is classification only (requires class boundaries).  

### Sampling-based, Clustering-based
- **Cross Entropy-based Sampling (CES), DeepReduce (DR), Practical Accuracy Estimation (PACE), Multiple-Boundary Clustering & Prioritization (MCP)** — See code.
- **DeepEST (EST)**  
  - *Classification (MNIST, AndroZoo, IMDb)*: **DSA + confidence** as auxiliary variable (best performance reported in the original paper).  
  - *Regression (Udacity)*: no softmax & DSA undefined ⇒ use **LSA** as auxiliary variable.  

### Hybrid
- **Distribution-Aware Test Selection (DAT)**  
  **Implementation of OOD detectors:** We utilize existing pre-trained models to conduct semi-supervised, distance-based OOD detection by using intermediate features, which is widely used in the literature due to its simplicity and effectiveness [17], [18]. We compute the Euclidean distance to the centroid of ID features from the training set. Samples with distances exceeding 95\% thresholds value are classified as OOD. We report the AUC-ROC score for performance check.
  - *MNIST*: **2nd hidden layer of LeNet-1** as the feature extractor. ;  **AUC-ROC 85.09%** with **Fashion-MNIST** as OOD.  
  - *AndroZoo*: **2nd hidden layer of DeepDrebin** as the feature extractor; **AUC-ROC 99.78%** with **FGSM** adversarial samples as OOD.  
  - *IMDb*: **raw, pre-processed text features**; **AUC-ROC 70.89%** with corrupted text as OOD.
  - *Udacity*: Not applicable.


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
[17] Y. Sun, Y. Ming, X. Zhu, and Y. Li, “Out-of-distribution detection with deep nearest neighbors,” in International conference on machine learning. PMLR, 2022, pp. 20 827–20 840.
[18] L. Ruff, R. Vandermeulen, N. Goernitz, L. Deecke, S. A. Siddiqui, A. Binder, E. M¨uller, and M. Kloft, “Deep one-class classification,” in International conference on machine learning. PMLR, 2018, pp.4393–4402.
