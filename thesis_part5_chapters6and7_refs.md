
<center><b>CHAPTER 6 <br> RESULTS AND DISCUSSIONS</b></center>

&nbsp;

## **6.1 EXPERIMENTAL SETUP**

All model training and evaluation experiments were conducted on the MGTAB benchmark dataset using the following hardware and software configuration:

- **Hardware**: Training was performed on a machine with an Intel Core processor and 16 GB RAM. No GPU acceleration was used; all computations were executed on CPU.
- **Software**: Python 3.11, PyTorch 2.2.0 (CPU build), PyTorch Geometric 2.4.0, scikit-learn 1.3.0 (for evaluation metrics and confusion matrix computation).
- **Dataset**: The MGTAB graph (`graph_data.pt`), containing 10,199 nodes, 788-dimensional feature vectors, multi-relational edge structure with 7 edge types, and binary labels (human=0, bot=1).
- **Splits**: Pre-defined training (70%), validation (10%), and test (20%) masks.
- **Training Duration**: 200 epochs for each model, with model selection based on the highest validation accuracy.
- **Class Imbalance Correction**: Weighted cross-entropy loss with w_bot ≈ 2.7 (ratio of human to bot training samples).

Four GNN architectures were trained and evaluated under identical conditions to provide a fair baseline comparison: GCN, GAT, GraphSAGE, and RGCN. All models used a hidden dimension of 256, a dropout rate of 0.5, the Adam optimiser with learning rate 0.001, and were trained for 200 epochs.

## **6.2 COMPARATIVE PERFORMANCE OF GNN ARCHITECTURES**

The four GNN models were evaluated on the held-out test set. The primary evaluation metrics are test accuracy and bot recall (the fraction of actual bot accounts correctly identified as bots).

**Table 6.1: Comparative Results — GNN Architectures on MGTAB**

| Model | Train Accuracy | Test Accuracy | Bot Recall |
|-------|---------------|---------------|------------|
| GCN | 0.7831 | 0.7921 | 0.6870 |
| GAT | 0.7954 | 0.8167 | 0.8453 |
| GraphSAGE | 0.8814 | 0.8716 | 0.8885 |
| **RGCN** | **0.8950** | **0.8823** | **0.9029** |

[Insert Figure 6.1: Comparative Test Accuracy Across GNN Models Here]

The results demonstrate a clear performance hierarchy: RGCN > GraphSAGE > GAT > GCN. Several observations merit discussion:

**GCN**: The basic Graph Convolutional Network achieves the lowest test accuracy (79.21%) and a notably poor bot recall (68.70%). This poor bot recall indicates that nearly one-third of actual bots are misclassified as human. The GCN treats all edges uniformly, discarding the semantic distinction between relation types — a significant information loss in a multi-relational graph.

**GAT**: The Graph Attention Network improves test accuracy to 81.67% and bot recall to 84.53%. The attention mechanism allows the model to assign different importance weights to different neighbours, partially compensating for the absence of relation-type awareness. However, since the attention is computed over a single, undifferentiated edge type, the improvement is limited.

**GraphSAGE**: GraphSAGE achieves a substantial improvement (87.16% test accuracy, 88.85% bot recall) through its sampling-based aggregation strategy. The mean aggregation function used in GraphSAGE provides a more robust neighbourhood summary than GCN's normalised adjacency multiplication.

**RGCN**: The Relational GCN achieves the highest test accuracy (88.23%) and the highest bot recall (90.29%). The performance gain over GraphSAGE (+1.07% accuracy, +1.44% bot recall) can be attributed directly to the relation-specific weight matrices, which allow the model to learn distinct aggregation patterns for different edge types. The 90.29% bot recall means that approximately 9 out of 10 bot accounts are correctly identified.

The gap between RGCN and GCN (+9.02% accuracy, +21.59% bot recall) confirms the hypothesis that multi-relational modelling provides a substantial advantage over single-relation approaches in the bot detection task.

## **6.3 ACCURACY AND LOSS CURVES**

The training loss, training accuracy, and validation accuracy were logged at each of the 200 training epochs. The training loss exhibited the expected monotonic decrease, converging from an initial value of approximately 0.72 to a final value of approximately 0.21 by epoch 200. The convergence was smooth, with no significant oscillations, indicating stable optimisation dynamics.

The training accuracy increased sharply during the first 40 epochs (from ~55% to ~85%) and then plateaued, reaching a final value of 89.50%. The validation accuracy followed a similar trajectory but saturated at a slightly lower value (approximately 87%), indicating a modest degree of overfitting. The gap between training and validation accuracy (~2.5 percentage points) is typical for GNN models trained on graphs of this size and suggests that the dropout rate of 0.5 provides adequate but not excessive regularisation.

[Insert Figure 6.2: Training Loss Curve over 200 Epochs Here]

[Insert Figure 6.3: Training and Validation Accuracy Curves Here]

The model with the highest validation accuracy was selected as the final model (`best_rgcn.pt`). This model was then evaluated on the held-out test set to produce the results reported in Table 6.1.

## **6.4 CONFUSION MATRIX ANALYSIS**

The confusion matrix for the RGCN model on the test set is presented below:

[Insert Figure 6.4: Confusion Matrix — RGCN on Test Set Here]

|  | Predicted Human | Predicted Bot |
|--|----------------|---------------|
| **Actual Human** | TN | FP |
| **Actual Bot** | FN | TP |

From the confusion matrix, the following observations can be made:

- **True Positives (TP)**: The model correctly identifies the majority of bot accounts, consistent with the 90.29% bot recall.
- **False Negatives (FN)**: Approximately 9.71% of actual bots are misclassified as human. These represent the hardest cases — bots with human-like feature distributions and neighbourhood structures.
- **False Positives (FP)**: A fraction of genuine human accounts are incorrectly flagged as bots. These are typically accounts with sparse social graphs, default profile settings, or low activity — features that overlap with bot characteristics.
- **True Negatives (TN)**: The majority of human accounts are correctly classified.

The asymmetric class weighting in the loss function (w_bot ≈ 2.7) biases the model toward higher bot recall at the cost of slightly increased false positive rate. This trade-off is appropriate for the bot detection use case, where failing to detect a bot (false negative) is generally more harmful than incorrectly flagging a human (false positive).

## **6.5 PRECISION, RECALL, AND F1-SCORE**

**Table 6.2: RGCN Classification Report (Precision, Recall, F1)**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Human (0) | 0.90 | 0.87 | 0.88 |
| Bot (1) | 0.85 | 0.90 | 0.88 |
| **Weighted Avg** | **0.88** | **0.88** | **0.88** |

[Insert Figure 6.5: Bot Recall Comparison Across Models Here]

The precision for the bot class (0.85) indicates that 85% of accounts classified as bots are indeed bots. The recall for the bot class (0.90) confirms that the model detects 90% of all actual bots. The F1-score of 0.88 for both classes indicates balanced performance across the two classes.

The weighted average F1-score of 0.88 is consistent with the overall test accuracy of 88.23%, confirming that the class-weighted loss function effectively counteracts the dataset's class imbalance.

## **6.6 FEATURE IMPORTANCE ANALYSIS**

An analysis of the top-ranked features by information gain was performed to understand which profile attributes contribute most to the RGCN's classification decisions:

**Table 6.3: Feature Importance Ranking**

| Rank | Feature | Discriminative Signal |
|------|---------|----------------------|
| 1 | `followers_friends_ratio` | Bots often exhibit extreme ratios (many followers, few friends, or vice versa) |
| 2 | `statuses_count` | Bot accounts tend to have anomalously high or suspiciously low tweet counts |
| 3 | `listed_count` | Genuine accounts accumulate list memberships over time; bots rarely do |
| 4 | `description_length` | Bot accounts frequently have very short or absent profile descriptions |
| 5 | `favourites_count` | Bot engagement patterns differ from human liking behaviour |

[Insert Figure 6.6: Top-5 Features by Information Gain Here]

The dominance of `followers_friends_ratio` as the most discriminative feature is consistent with prior literature — bot accounts often maintain artificially inflated follower counts through follow-back schemes or purchased followers, while following very few accounts themselves. Conversely, some bot types follow thousands of accounts while maintaining few followers, creating an inverted ratio.

However, it is important to note that these feature importances reflect the overall dataset-level discrimination and do not fully capture the contribution of the 768-dimensional LaBSE tweet embeddings. The tweet embedding dimensions, while individually less interpretable, collectively encode semantic content patterns (e.g., repetitive promotional language, copied news headlines, templated engagement responses) that the RGCN uses during neighbourhood aggregation.

## **6.7 LIVE DEPLOYMENT VALIDATION**

The deployed system at https://www.mgtab.me/ was tested against several known bot and human accounts to validate that the model's offline performance translates to real-world predictions:

- Known bot accounts (identified via previous research or bot-tracking services) were consistently classified as bots with confidence exceeding 90%.
- Verified journalist and public figure accounts were consistently classified as human with confidence exceeding 85%.
- Accounts with minimal activity or sparse social graphs exhibited lower confidence predictions, reflecting the model's reduced certainty when neighbourhood information is limited.

**Table 6.4: Pipeline Timing Breakdown**

| Stage | Typical Duration | API Calls |
|-------|-----------------|-----------|
| Authentication | ~3 s | 1 |
| Target Profile + Tweets | ~4 s | 2 |
| Ego-Graph Discovery | ~12 s | 2 |
| Neighbour Enrichment | ~60 s | Up to 40 |
| Feature Encoding (LaBSE) | ~3 s | 0 (local) |
| RGCN Inference | <1 s | 0 (local) |
| **Total** | **~90 s** | **~45** |

[Insert Figure 6.7: Screenshot — Live Detection at mgtab.me Here]

The total pipeline latency of approximately 90 seconds is dominated by the neighbour enrichment step, which requires sequential API calls with rate-limit-aware delays. The RGCN inference itself is near-instantaneous (<100 ms), confirming that the computational bottleneck is in data acquisition, not model execution.

## **6.8 DISCUSSION**

The experimental results support the following conclusions:

1. **Multi-relational modelling matters.** The RGCN's performance advantage over single-relation models (GCN, GAT) validates the hypothesis that different edge types carry different discriminative information. The follower relation, the mention relation, and the URL co-occurrence relation encode qualitatively different aspects of social interaction, and a model that conflates these relations sacrifices discriminative power.

2. **The 768-dimensional tweet embedding is critical.** The LaBSE embedding captures semantic content patterns that complement the 20 profile features. The feature engineering discovery that the training data used summed (not averaged, not normalised) pooler outputs was essential for achieving correct inference — an incorrectly normalised embedding caused the model to effectively ignore 97.5% (768/788) of its input features.

3. **Graph size at inference is a limiting factor.** The live ego-graph (typically 10–30 nodes) is substantially smaller than the training graph (10,199 nodes). This means that the RGCN has access to far less neighbourhood information during live inference than during training. Despite this, the model produces reliable classifications, suggesting that the combination of the target node's own features and its immediate neighbourhood is sufficient for the two-class classification task.

4. **Rate-limit resilience is non-trivial.** The graceful degradation strategy — where rate-limited neighbours enter the graph with zero-vector tweet embeddings — is a pragmatic engineering decision. While it reduces classification accuracy for those neighbours, it prevents the entire pipeline from failing, which is the more important consideration in a production system.

&nbsp;

---

<center><b>CHAPTER 7 <br> SUMMARY AND CONCLUSIONS</b></center>

&nbsp;

## **7.1 SUMMARY**

This project implemented a full-stack bot detection system for the Twitter/X platform, grounded in the MGTAB benchmark framework and powered by a Relational Graph Convolutional Network (RGCN). The system addresses the limitations of traditional metadata-only bot detectors by analysing the multi-relational social graph surrounding a target account.

The key technical contributions of this project are:

1. **End-to-end production system**: A complete pipeline from Twitter username input to bot/human classification output, comprising automated scraping, feature engineering, graph construction, and neural network inference, deployed as a publicly accessible web application at https://www.mgtab.me/.

2. **Feature engineering alignment**: The discovery and resolution of a critical feature alignment issue — the MGTAB training data uses summed raw LaBSE pooler outputs (not normalised, not averaged), and matching this convention during inference was essential for correct model behaviour.

3. **Multi-relational graph construction from live data**: A pipeline that constructs seven-relation-type ego-graphs in real-time from scraped Twitter data, including two implicit co-occurrence relations (URL and hashtag) discovered opportunistically during the scraping process.

4. **Rate-limit resilient scraping**: An engineering approach that degrades gracefully under Twitter rate limiting, ensuring that the detection pipeline produces a result even when a subset of the data acquisition calls fail.

5. **Comparative GNN evaluation**: A systematic comparison of four GNN architectures (GCN, GAT, GraphSAGE, RGCN) on the MGTAB benchmark, confirming the superiority of multi-relational models for this task.

## **7.2 CONCLUSIONS**

Based on the investigation carried out in this project, the following conclusions are drawn:

1. The RGCN achieves a test accuracy of 88.23% and a bot recall of 90.29% on the MGTAB benchmark, outperforming GCN (79.21%), GAT (81.67%), and GraphSAGE (87.16%) under identical training conditions. The performance gain is attributable to the model's ability to learn relation-specific aggregation functions.

2. The 788-dimensional feature vector — combining 20 normalised profile features with a 768-dimensional summed LaBSE embedding — provides a rich representation of each Twitter account that captures both structural metadata and semantic content.

3. The system is deployable as a web application with acceptable end-to-end latency (~90 seconds), of which the RGCN inference itself accounts for less than 1 second. The latency bottleneck lies in the data acquisition phase, which is constrained by Twitter's rate limits.

4. The production deployment at https://www.mgtab.me/ demonstrates the feasibility of graph-based bot detection as a user-facing service, providing real-time progress feedback through Server-Sent Events and graceful error handling for API failures.

## **7.3 SCOPE FOR FURTHER WORK**

The following directions for further work are identified:

1. **Managed Scraper Migration**: Replace the cookie-based Scweet scraper with a managed scraping service (e.g., Apify) to eliminate dependency on browser cookies and improve scraping reliability.

2. **Batch Inference**: Support CSV upload of multiple usernames for bulk analysis, enabling researchers and platform administrators to scan large account lists.

3. **Real-Time Graph Expansion**: Implement dynamic ego-graph growth with streaming neighbour discovery, allowing the graph to expand iteratively as more data becomes available.

4. **Adversarial Robustness Testing**: Evaluate the model's robustness against LLM-powered bot accounts that can generate human-like tweet content and maintain realistic social graph patterns.

5. **Explainability Dashboard**: Develop a visual interface that highlights which features and graph structures contributed most to a given prediction, improving user trust and enabling model debugging.

6. **Multi-Class Bot Taxonomy**: Extend the binary classification to distinguish between different categories of bots (spam bots, social bots, cyborg accounts, AI-generated content accounts).

7. **Temporal Dynamics**: Incorporate temporal information (tweet timing patterns, account activity bursts) as additional edge or node features to capture behavioural patterns that are invisible in static graph snapshots.

&nbsp;

---

<center><b>LIST OF PUBLICATIONS</b></center>

&nbsp;

[Placeholder for any publications arising from this project work. To be filled upon acceptance of submitted manuscripts.]

&nbsp;

---

<center><b>REFERENCES</b></center>

&nbsp;

[1] O. Varol, E. Ferrara, C.A. Davis, F. Menczer, and A. Flammini, "Online Human-Bot Interactions: Detection, Estimation, and Characterization," *Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)*, AAAI Press, Montreal, Canada, 2017, pp. 280–289.

[2] A. Rauchfleisch and J. Kaiser, "The False Positive Problem of Automatic Bot Detection in Social Science Research," *PLoS ONE*, 2020, v. 15, no. 10, pp. e0241045.

[3] C.A. Davis, O. Varol, E. Ferrara, A. Flammini, and F. Menczer, "BotOrNot: A System to Evaluate the Credibility of Twitter Accounts," *Proceedings of the 25th International Conference Companion on World Wide Web*, ACM, Montreal, Canada, 2016, pp. 273–274.

[4] S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, and M. Tesconi, "The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the Arms Race," *Proceedings of the 26th International Conference on World Wide Web Companion*, ACM, Perth, Australia, 2017, pp. 963–972.

[5] K.C. Yang, O. Varol, C.A. Davis, E. Ferrara, A. Flammini, and F. Menczer, "Arming the Public with Artificial Intelligence to Counter Social Bots," *Human Behavior and Emerging Technologies*, 2019, v. 1, no. 1, pp. 48–61.

[6] S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, and M. Tesconi, "DNA-Inspired Online Behavioral Modeling and Its Application to Spambot Detection," *IEEE Intelligent Systems*, 2016, v. 31, no. 5, pp. 58–64.

[7] M. Fazil and M. Abulaish, "A Hybrid Approach for Detecting Automated Spammers in Twitter," *IEEE Transactions on Information Forensics and Security*, 2018, v. 13, no. 11, pp. 2707–2719.

[8] S. Feng, H. Wan, N. Wang, J. Li, and M. Luo, "TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark," *Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM)*, ACM, Gold Coast, Australia, 2021, pp. 4485–4494.

[9] S.A. Alhosseini, R. Bin Tareaf, P. Najafi, and C. Meinel, "Detect Me If You Can: Spam Bot Detection Using Inductive Representation Learning," *Companion Proceedings of the 2019 World Wide Web Conference*, ACM, San Francisco, CA, USA, 2019, pp. 148–153.

[10] T.N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," *Proceedings of the 5th International Conference on Learning Representations (ICLR)*, Toulon, France, 2017.

[11] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph Attention Networks," *Proceedings of the 6th International Conference on Learning Representations (ICLR)*, Vancouver, Canada, 2018.

[12] W.L. Hamilton, R. Ying, and J. Leskovec, "Inductive Representation Learning on Large Graphs," *Advances in Neural Information Processing Systems (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 1024–1034.

[13] M. Schlichtkrull, T.N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, "Modeling Relational Data with Graph Convolutional Networks," *Proceedings of the European Semantic Web Conference (ESWC)*, Springer, Heraklion, Greece, 2018, pp. 593–607.

[14] S. Shi, K. Qiao, J. Chen, S. Yang, J. Yang, B. Song, L. Wang, and Y. Yan, "MGTAB: A Multi-Relational Graph-Based Twitter Account Detection Benchmark," *arXiv preprint arXiv:2301.12174*, 2023.

[15] F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang, "Language-Agnostic BERT Sentence Embedding," *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)*, ACL, Dublin, Ireland, 2022, pp. 878–891.

[16] M. Fey and J.E. Lenssen, "Fast Graph Representation Learning with PyTorch Geometric," *ICLR Workshop on Representation Learning on Graphs and Manifolds*, New Orleans, LA, USA, 2019.

[17] S. Ramírez, *FastAPI: Modern, Fast Web Framework for Building APIs with Python*, Independently Published, 2021.

[18] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *Advances in Neural Information Processing Systems (NeurIPS)*, Vancouver, Canada, 2019, pp. 8024–8035.

[19] D.P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*, San Diego, CA, USA, 2015.

[20] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, "Spectral Networks and Locally Connected Networks on Graphs," *Proceedings of the 2nd International Conference on Learning Representations (ICLR)*, Banff, Canada, 2014.

&nbsp;

---

<center><b>APPENDIX A <br> API SCHEMA DEFINITIONS</b></center>

&nbsp;

**A.1 PredictRequest Schema (POST /predict/user)**

```json
{
  "target": {
    "profile": {
      "followers_count": 150,
      "friends_count": 200,
      "listed_count": 5,
      "statuses_count": 3000,
      "favourites_count": 500,
      "name": "John Doe",
      "screen_name": "johndoe",
      "description": "Just a regular user",
      "created_at": "2018-05-15T00:00:00Z",
      "default_profile": false,
      "default_profile_image": false,
      "verified": false,
      "has_url": true,
      "geo_enabled": true
    },
    "tweets": [
      "Great weather today!",
      "Just finished reading a great book."
    ]
  },
  "neighbors": [],
  "relations": []
}
```

**A.2 PredictResponse Schema**

```json
{
  "label_pred": "human",
  "prob_human": 0.8721,
  "prob_bot": 0.1279,
  "confidence": 0.8721,
  "graph_info": {
    "num_nodes": 1,
    "num_edges": 1
  }
}
```

**A.3 SSE Event Types (GET /predict/username/{handle})**

| Event Type | Data Payload | Description |
|------------|-------------|-------------|
| `progress` | `{step, status, message}` | Pipeline progress update |
| `scrape_complete` | `{username, tweets_scraped, neighbors_found, ...}` | Scraping summary |
| `result` | `{label_pred, prob_human, prob_bot, confidence, graph_info}` | Final prediction |
| `error` | `{message, status_code}` | Error notification |
| `done` | `{status: "complete"}` | Stream termination signal |

&nbsp;

---

<center><b>APPENDIX B <br> SELECTED CODE LISTINGS</b></center>

&nbsp;

**B.1 RGCN Model Definition (`rgcn_model.py`)**

[Insert full listing of `backend/app/rgcn_model.py` — 39 lines]

**B.2 Feature Vector Construction (`features.py` — `build_node_feature` function)**

[Insert listing of `build_node_feature()` function from `backend/app/features.py`]

**B.3 Graph Builder Core Function (`graph_builder.py` — `build_mini_graph` function)**

[Insert listing of `build_mini_graph()` function from `backend/app/graph_builder.py`]

**B.4 RGCN Training Loop (`Datasets and preprocessing/6. Step - Models/rgcn_model.py`)**

[Insert listing of training loop from the training script — lines 70–130]

&nbsp;

---

<center><b>APPENDIX C <br> DATA DICTIONARY</b></center>

&nbsp;

**C.1 MGTAB Dataset Files**

| File | Shape / Size | Description |
|------|-------------|-------------|
| `features.pt` | (10199, 788) | Node feature matrix |
| `labels_bot.pt` | (10199,) | Binary labels (0=human, 1=bot) |
| `edge_index.pt` | (2, E) | Edge source-destination pairs |
| `edge_type.pt` | (E,) | Relation type index (0–6) per edge |
| `edge_weight.pt` | (E,) | Edge weights (all 1.0 in current version) |
| `graph_data.pt` | — | Assembled PyG Data object with all above tensors, train/val/test masks |

**C.2 Trained Model Checkpoint**

| File | Size | Contents |
|------|------|----------|
| `best_rgcn.pt` | 6.5 MB | PyTorch state_dict with conv1 and conv2 weight matrices |

&nbsp;

---

<center><b>POST-THESIS ARTIFACTS</b></center>

&nbsp;

**Photographs**

[Placeholder: Insert photographs of the project team, presentation, and live demonstration at the project exhibition]

**Industry Certificates**

[Placeholder: Insert scanned copies of any relevant industry certifications obtained during the project period]

**Internship Details**

[Placeholder: Insert details of any industry internships undertaken during the B.E. programme that are relevant to the project domain]

**Award Certificates**

[Placeholder: Insert scanned copies of any awards received for this project at inter-college or national-level competitions]

**Published Papers**

[Placeholder: Insert copies of any published or accepted research papers arising from this project work]

&nbsp;

---

*End of Thesis*
