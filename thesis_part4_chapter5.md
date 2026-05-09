
<center><b>CHAPTER 5 <br> RGCN MODEL: MATHEMATICAL FOUNDATIONS AND IMPLEMENTATION</b></center>

&nbsp;

## **5.1 GRAPH CONVOLUTIONAL NETWORKS — BACKGROUND**

Before presenting the RGCN architecture used in this project, it is necessary to establish the mathematical framework of Graph Convolutional Networks (GCNs) from which RGCNs are derived.

Consider an undirected graph G = (V, E) with N = |V| nodes and E = |E| edges. Each node i ∈ V is associated with a feature vector **x**_i ∈ ℝ^d. The graph structure is encoded in the adjacency matrix **A** ∈ ℝ^(N×N), where A_{ij} = 1 if there exists an edge between nodes i and j, and A_{ij} = 0 otherwise.

The spectral convolution on graphs, as defined by Bruna et al. (2014), expresses the convolution of a signal **x** with a filter g_θ in the spectral domain of the graph Laplacian:

g_θ ⋆ **x** = **U** g_θ(Λ) **U**^T **x**

where **U** is the matrix of eigenvectors of the normalised graph Laplacian **L** = **I** - **D**^(-1/2) **A** **D**^(-1/2) = **U** Λ **U**^T, and Λ is the diagonal matrix of eigenvalues. Computing this expression requires O(N²) operations and the full eigendecomposition of **L**, which is prohibitively expensive for large graphs.

Kipf and Welling (2017) approximated this spectral convolution using a first-order Chebyshev polynomial expansion, yielding the GCN layer-wise propagation rule:

**H**^(l+1) = σ( **D̃**^(-1/2) **Ã** **D̃**^(-1/2) **H**^(l) **W**^(l) )

where:
- **Ã** = **A** + **I**_N is the adjacency matrix with added self-loops,
- **D̃** is the diagonal degree matrix of **Ã**, with D̃_{ii} = Σ_j Ã_{ij},
- **H**^(l) ∈ ℝ^(N×d_l) is the matrix of node representations at layer l (with **H**^(0) = **X**, the input feature matrix),
- **W**^(l) ∈ ℝ^(d_l × d_{l+1}) is the trainable weight matrix at layer l, and
- σ is a non-linear activation function, typically ReLU(x) = max(0, x).

The normalised adjacency multiplication **D̃**^(-1/2) **Ã** **D̃**^(-1/2) **H**^(l) performs a weighted average of each node's feature vector with its neighbours' feature vectors. This operation is equivalent to a single round of *message passing*: each node "receives" information from its immediate graph neighbours. Stacking L layers allows each node to aggregate information from nodes up to L hops away.

The computational complexity of a single GCN layer is O(E × d_{l+1}) for the sparse matrix multiplication and O(N × d_l × d_{l+1}) for the weight matrix multiplication.

[Insert Figure 5.1: Standard GCN Message-Passing (Single Relation) Here]

**Limitation in the Multi-Relational Setting.** The GCN propagation rule applies a *single* weight matrix **W**^(l) uniformly to all edges. When the graph contains edges of semantically different types (e.g., follower, mention, reply), a single weight matrix cannot learn type-specific transformations. A follower edge and a reply edge carry fundamentally different information about the relationship between two accounts, and conflating them into a single aggregation operation discards this distinction.

## **5.2 RELATIONAL GRAPH CONVOLUTIONAL NETWORKS**

### **5.2.1 Message-Passing Formulation**

The Relational Graph Convolutional Network (RGCN), introduced by Schlichtkrull et al. (2018), extends the GCN framework to multi-relational graphs. A multi-relational graph is defined as G = (V, E, R), where R is the set of relation types and each edge (i, j, r) ∈ E connects nodes i and j with relation type r ∈ R.

The RGCN message-passing rule for node i at layer l is:

**h**_i^(l+1) = σ( **W**_0^(l) **h**_i^(l) + Σ_{r∈R} Σ_{j∈N_i^r} (1 / c_{i,r}) **W**_r^(l) **h**_j^(l) )

where:
- **h**_i^(l) ∈ ℝ^(d_l) is the hidden representation of node i at layer l,
- N_i^r is the set of neighbours of node i connected via relation type r,
- **W**_r^(l) ∈ ℝ^(d_{l+1} × d_l) is the weight matrix specific to relation type r at layer l,
- **W**_0^(l) ∈ ℝ^(d_{l+1} × d_l) is the self-loop weight matrix at layer l,
- c_{i,r} is a normalisation constant, typically |N_i^r| (the number of neighbours under relation r),
- σ is a non-linear activation function.

The summation structure of this formula can be decomposed into three conceptual components:

1. **Self-connection**: **W**_0^(l) **h**_i^(l) — the node's own features are linearly transformed and contribute to its updated representation. This term ensures that the node retains its own information across layers.

2. **Relation-specific aggregation**: For each relation type r, the features of all neighbours connected via relation r are averaged (normalised by c_{i,r}) and linearly transformed by the relation-specific weight matrix **W**_r^(l). This is the core innovation — different relation types induce different linear transformations.

3. **Cross-relation summation**: The contributions from all relation types are summed. The model is thereby able to combine signals from different types of social interactions.

[Insert Figure 5.2: RGCN Message-Passing with Relation-Specific Weights Here]

In the context of the MGTAB bot detection task, this formulation means:
- The model learns a separate weight matrix for follower relationships (**W**_follower), friend relationships (**W**_friend), mention relationships (**W**_mention), reply relationships (**W**_reply), quoted relationships (**W**_quoted), URL co-occurrence relationships (**W**_url), and hashtag co-occurrence relationships (**W**_hashtag).
- When updating the representation of a target node, the model computes: "What do my followers look like (via **W**_follower)? What do the accounts I follow look like (via **W**_friend)? What do the accounts I mention look like (via **W**_mention)?" — and combines these answers with different learned linear transformations.

### **5.2.2 Basis Decomposition**

Maintaining |R| separate weight matrices, each of dimension d_l × d_{l+1}, introduces O(|R| × d_l × d_{l+1}) parameters per layer. For the first layer of our model (d_0 = 788, d_1 = 256, |R| = 7), this amounts to 7 × 788 × 256 = 1,412,096 parameters — which is manageable but may lead to overfitting on smaller datasets.

Schlichtkrull et al. proposed two regularisation strategies:

**Basis Decomposition**: Each relation-specific weight matrix is expressed as a linear combination of B shared basis matrices:

**W**_r^(l) = Σ_{b=1}^{B} a_{rb}^(l) **V**_b^(l)

where **V**_b^(l) ∈ ℝ^(d_{l+1} × d_l) are shared basis matrices and a_{rb}^(l) ∈ ℝ are relation-specific scalar coefficients. This reduces the parameter count to O(B × d_l × d_{l+1} + |R| × B).

**Block-Diagonal Decomposition**: Each **W**_r^(l) is constrained to be block-diagonal, reducing the number of parameters per matrix from d_l × d_{l+1} to B × (d_l / B) × (d_{l+1} / B).

The PyTorch Geometric implementation of `RGCNConv` used in this project employs the default approach without explicit basis decomposition (num_bases=None), as the number of relation types (7) is small enough that the full parameterisation does not cause overfitting on the 10,199-node MGTAB dataset.

### **5.2.3 Layer-wise Propagation Rule**

For the specific two-layer RGCN used in this project, the complete forward computation is:

**Layer 1:**

**h**_i^(1) = ReLU( **W**_0^(0) **x**_i + Σ_{r=0}^{6} Σ_{j∈N_i^r} (1 / |N_i^r|) **W**_r^(0) **x**_j )

where **x**_i ∈ ℝ^788 is the input feature vector and **h**_i^(1) ∈ ℝ^256.

**Dropout:**

**h**_i^(1) = Dropout(**h**_i^(1), p=0.5)

During training, each element of the 256-dimensional hidden vector is independently set to zero with probability 0.5, and the remaining elements are scaled by 1/(1−0.5) = 2. During inference, dropout is disabled, and the full hidden vector is used.

**Layer 2:**

**z**_i = **W**_0^(1) **h**_i^(1) + Σ_{r=0}^{6} Σ_{j∈N_i^r} (1 / |N_i^r|) **W**_r^(1) **h**_j^(1)

where **z**_i ∈ ℝ^2 are the raw logits (unnormalised scores) for node i. No activation function is applied after the final layer.

**Classification:**

P(y_i = k) = softmax(**z**_i)_k = exp(z_{i,k}) / Σ_{k'=0}^{1} exp(z_{i,k'})

The predicted class is ŷ_i = argmax_k P(y_i = k), and the confidence is max_k P(y_i = k).

## **5.3 MODEL ARCHITECTURE**

The RGCN model is implemented in `rgcn_model.py` as a PyTorch `nn.Module` subclass with 39 lines of code. The architecture is defined as:

```
Input: x ∈ ℝ^(N×788), edge_index ∈ ℤ^(2×E), edge_type ∈ ℤ^E
    │
    ▼
RGCNConv Layer 1: 788 → 256 (num_relations=7)
    │
    ▼
ReLU Activation
    │
    ▼
Dropout (p=0.5, training only)
    │
    ▼
RGCNConv Layer 2: 256 → 2 (num_relations=7)
    │
    ▼
Output: logits ∈ ℝ^(N×2)
```

[Insert Figure 5.3: RGCN Model Architecture (788 → 256 → 2) Here]

The model uses the `RGCNConv` operator from the PyTorch Geometric library, which implements the message-passing formulation described in Section 5.2.1. The operator handles sparse matrix operations, relation-specific weight management, and normalisation internally.

The total number of trainable parameters is:

- Layer 1: 8 weight matrices (7 relation + 1 self-loop) × 788 × 256 = 1,613,824 parameters + 256 bias = 1,614,080
- Layer 2: 8 weight matrices × 256 × 2 = 4,096 parameters + 2 bias = 4,098
- **Total: ~1,618,178 parameters** (~6.2 MB when serialised as float32)

## **5.4 TRAINING PROCEDURE**

### **5.4.1 Loss Function and Class Imbalance Correction**

The MGTAB dataset exhibits a class imbalance of approximately 2.3:1 (human:bot). If trained with a standard cross-entropy loss, the model would be biased toward predicting the majority class (human), resulting in low bot recall — which is unacceptable for a bot detection system where missing bots is more costly than false positives.

To address this, a weighted cross-entropy loss is used:

L = - (1/N) Σ_{i∈train} w_{y_i} log P(y_i | **z**_i)

where the class weights are:
- w_human = 1.0
- w_bot = N_human / N_bot ≈ 2.7

The weight for the bot class is set to the ratio of human to bot samples in the training set, effectively up-weighting the gradient contribution of misclassified bot samples. This is implemented using PyTorch's `nn.CrossEntropyLoss(weight=class_weights)`.

### **5.4.2 Optimiser and Hyperparameters**

**Table 5.1: RGCN Hyperparameter Summary**

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 (default) |
| Dropout rate | 0.5 |
| Hidden dimension | 256 |
| Number of RGCN layers | 2 |
| Number of relation types | 7 |
| Number of training epochs | 200 |
| Batch size | Full-batch (entire graph) |
| Model selection criterion | Best validation accuracy |

The training is performed in full-batch mode — the entire graph (10,199 nodes) is processed in each forward pass. This is feasible because the MGTAB graph fits comfortably in CPU memory. The Adam optimiser is used with its default β₁ = 0.9, β₂ = 0.999, ε = 10^(-8).

Model selection follows the standard early stopping protocol: the model state with the highest validation accuracy across all 200 epochs is saved as `best_rgcn.pt`.

[Insert Figure 5.4: Training Pipeline Flowchart Here]

### **5.4.3 Dataset Splits**

The MGTAB dataset is split into training, validation, and test sets using the pre-defined masks provided in the `graph_data.pt` file:

**Table 5.2: MGTAB Dataset Split Statistics**

| Split | Nodes | Purpose |
|-------|-------|---------|
| Training | ~7,140 (70%) | Weight optimisation |
| Validation | ~1,020 (10%) | Hyperparameter tuning and model selection |
| Test | ~2,039 (20%) | Final evaluation (reported metrics) |

The splits are applied via boolean masks (`train_mask`, `val_mask`, `test_mask`) that index into the full node set. During training, the loss is computed only over nodes in the training mask, but the forward pass operates on the entire graph — this is standard practice for transductive GNN training, as it allows message passing to utilise information from validation and test nodes' features (but not their labels).

## **5.5 INFERENCE PIPELINE**

The inference pipeline, implemented in `inference.py`, operates as follows:

1. **Model Loading** (at server startup): The `InferenceEngine` class instantiates the RGCN model with the same architecture used during training, loads the saved state dictionary from `best_rgcn.pt`, sets the model to evaluation mode (`model.eval()`), and places it on the CPU device.

2. **Graph Construction**: The `predict_from_request()` method receives the request JSON (containing target profile/tweets, neighbour data, and relations) and calls `build_mini_graph()` to construct the PyTorch Geometric `Data` object.

3. **Forward Pass**: The `predict()` method moves the `Data` object to the compute device (CPU), executes the model's forward pass with gradient computation disabled (`torch.no_grad()`), and extracts the logits for the target node (always at index 0).

4. **Softmax and Classification**: The target node's 2-dimensional logit vector is passed through the softmax function to obtain class probabilities. The predicted label is determined by `argmax`, and the confidence is the maximum probability.

5. **Result Assembly**: The prediction result is packaged as a JSON-serialisable dictionary containing `label_pred` (string: "human" or "bot"), `prob_human`, `prob_bot`, `confidence`, and `graph_info` (metadata about the mini-graph's size).

[Insert Figure 5.5: Inference Pipeline from Request to Prediction Here]

The inference time for the RGCN forward pass is consistently below 100 milliseconds on CPU, as the mini-graph used for live inference typically contains only 10–30 nodes and 20–80 edges — orders of magnitude smaller than the full training graph.

&nbsp;

---
