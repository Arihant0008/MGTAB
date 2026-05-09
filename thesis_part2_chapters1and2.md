
<center><b>CHAPTER 1 <br> INTRODUCTION</b></center>

&nbsp;

## **1.1 GENERAL**

Social media platforms, and Twitter (now rebranded as X) in particular, serve as critical channels for public discourse, political communication, brand engagement, and news dissemination. As of 2024, the platform hosts over 500 million active accounts, a non-trivial fraction of which are automated accounts commonly referred to as bots. These accounts range from benign automated services (weather alerts, news aggregation) to malicious entities engaged in coordinated inauthentic behaviour — amplifying misinformation, manipulating trending topics, inflating follower counts, and distorting public sentiment on issues of social and political significance.

The detection of such automated accounts has emerged as a well-studied problem within the computational social science and network security communities. Early detection systems relied on straightforward heuristic rules: accounts posting at unusually high frequencies, accounts with disproportionate follower-to-friend ratios, or accounts with default profile images. However, the arms race between bot operators and detection systems has progressively rendered these heuristic rules insufficient. Modern bot accounts, equipped with generative language models and sophisticated scheduling algorithms, produce content and maintain activity patterns that are virtually indistinguishable from those of genuine human users when examined in isolation.

This project concerns itself with a fundamentally different approach to the bot detection problem — one that analyses not individual accounts in isolation, but the *relational structure* surrounding a target account. The working hypothesis, supported by the existing literature, is that while bot operators can fabricate convincing profile metadata and generate plausible tweet content, they cannot easily replicate the organic patterns found in authentic social network topology. The structural signatures of follower graphs, mention networks, reply chains, and content co-occurrence patterns carry discriminative signals that profile-level features alone do not capture.

The system described in this thesis implements the MGTAB (Multi-Relational Graph-Based Twitter Account Detection Benchmark) framework as a production-ready web application. The detection model at its core is a Relational Graph Convolutional Network (RGCN) — a graph neural network architecture specifically designed for multi-relational data — trained on a benchmark dataset of 10,199 expert-annotated Twitter accounts connected by seven distinct relation types. The application is deployed and publicly accessible at https://www.mgtab.me/.

## **1.2 PROBLEM STATEMENT**

Given a Twitter/X username as input, the system must determine whether the corresponding account is operated by a human or by an automated bot, using graph-based analysis of the account's social neighbourhood. The specific technical challenges addressed are:

1. **Data Acquisition Without Official API Access**: The Twitter API v2 imposes significant cost barriers ($100/month for basic access). The system must acquire profile metadata, tweet content, and social graph information through alternative means while remaining resilient to rate-limiting.

2. **Real-Time Graph Construction**: For each query, the system must construct a multi-relational ego-graph from live scraped data, incorporating up to seven relation types (follower, friend, mention, reply, quote, URL co-occurrence, hashtag co-occurrence), and encode each node as a 788-dimensional feature vector in near real-time.

3. **Multi-Relational Classification**: The detection model must exploit the heterogeneous nature of the constructed graph — specifically, it must learn separate relational patterns for each of the seven edge types rather than treating all edges uniformly.

4. **Production Deployment**: The system must be accessible as a web application with an intuitive user interface, with backend inference completing within a reasonable time window despite the computationally expensive scraping pipeline.

## **1.3 AIM AND OBJECTIVES**

The aim of this project is to design, implement, and deploy a web-based bot detection system that leverages multi-relational graph neural networks for classifying Twitter/X accounts as human or bot.

The specific objectives are as follows:

1. To study the MGTAB benchmark framework and reproduce its multi-relational graph structure using live data scraped from Twitter/X.

2. To implement and train an RGCN model on the MGTAB dataset comprising 10,199 nodes and seven relation types, and to evaluate its performance against baseline GNN architectures (GCN, GAT, GraphSAGE).

3. To develop a feature engineering pipeline that constructs a 788-dimensional node feature vector combining 20 normalised profile features and a 768-dimensional LaBSE tweet embedding.

4. To build a FastAPI backend that orchestrates asynchronous scraping, graph construction, feature encoding, and model inference, streaming real-time progress updates to the frontend via Server-Sent Events.

5. To develop a React 19 frontend that provides both a one-click automated analysis mode and a manual data-entry mode for fallback operation.

6. To deploy the complete system to a publicly accessible URL (https://www.mgtab.me/) using Hugging Face Spaces for the backend and Vercel for the frontend.

## **1.4 SCOPE OF THE PROJECT**

The scope of this project encompasses the following:

- **Detection Target**: Binary classification of Twitter/X accounts into two classes — human (class 0) and bot (class 1). Multi-class bot taxonomy (e.g., spam bots, social bots, cyborg accounts) is outside the current scope.

- **Graph Construction**: The system constructs ego-graphs centred on a single target user, with up to approximately 50 neighbouring nodes discovered through five explicit relation types (follower, friend, mention, reply, quote) and two implicit relation types (URL and hashtag co-occurrence).

- **Feature Space**: The node feature vector is fixed at 788 dimensions — 20 profile metadata features normalised using log-MinMax scaling, concatenated with a 768-dimensional LaBSE embedding obtained by summing the pooler outputs across a user's recent tweets.

- **Model Architecture**: A two-layer RGCN with a hidden dimension of 256 and seven relation types. The model weights are trained offline on the full MGTAB dataset and loaded at server startup; online learning is not supported.

- **Deployment**: The system is deployed as a web application. Batch processing of multiple usernames (e.g., via CSV upload) is not supported in the current version.

- **Platform**: The system is specifically designed for Twitter/X. Extension to other social media platforms (e.g., Reddit, Facebook) would require re-engineering the scraping pipeline and re-defining the relation types.

## **1.5 MOTIVATION FOR MULTI-GRAPH APPROACHES**

The motivation for adopting a multi-relational graph-based approach over traditional feature-based methods can be summarised along three dimensions:

**Limitation of Profile-Only Features.** Conventional bot detection systems extract a fixed set of features from each account's profile metadata — follower count, friend count, account age, tweet frequency, and similar attributes. While these features were historically sufficient to identify early-generation bots that exhibited obvious statistical anomalies (e.g., zero followers, thousands of tweets per day), contemporary bot operators have learned to maintain profile statistics that fall within the normal range for human accounts. A profile-only classifier trained on such features will increasingly produce false negatives as bot sophistication increases.

**Structural Inimitability.** The social graph surrounding a genuine human account reflects years of organic relationship formation — mutual follows among real-world acquaintances, mention chains arising from genuine conversations, reply threads reflecting topical interest, and shared URLs and hashtags that emerge from participation in authentic communities. Bot accounts, by contrast, tend to exhibit characteristic structural anomalies: followers drawn from disjoint, unrelated communities; mention patterns that suggest coordinated amplification campaigns; and content co-occurrence patterns that reflect automated topic injection rather than organic interest. These structural signatures are difficult to fabricate at scale.

**Information Aggregation Across Neighbours.** A graph neural network can aggregate information from a node's neighbourhood, effectively using the features and labels of connected accounts to inform the classification of the target node. If a target account's followers are predominantly other bots (which may be individually hard to classify), the graph neural network can leverage this neighbourhood composition as an additional discriminative signal. This neighbourhood-aware classification is fundamentally impossible in a feature-only framework.

[Insert Figure 1.2: Isolated vs. Graph-Based Account Analysis Here]

## **1.6 ORGANISATION OF THE THESIS**

The remainder of this thesis is organised as follows:

**Chapter 2** presents a review of the existing literature on bot detection, covering traditional feature-based approaches, graph-based methods, and graph neural network architectures, culminating in a discussion of the MGTAB benchmark.

**Chapter 3** describes the system architecture and design of the deployed application, detailing the React frontend component hierarchy, FastAPI backend route structure, and the deployment infrastructure.

**Chapter 4** covers the data ingestion pipeline and feature engineering methodology, including the Scweet-based scraping strategy, the 788-dimensional feature vector construction, and the multi-relational graph building process.

**Chapter 5** presents the mathematical formulation of the RGCN model, the training procedure, and the inference pipeline.

**Chapter 6** reports the experimental results, including comparative performance of GNN architectures on the MGTAB dataset, confusion matrix analysis, and discussion of live deployment validation.

**Chapter 7** concludes the thesis with a summary, conclusions, and a discussion of directions for further work.

&nbsp;

---

<center><b>CHAPTER 2 <br> REVIEW OF LITERATURE</b></center>

&nbsp;

## **2.1 FEATURE-BASED BOT DETECTION METHODS**

The earliest systematic efforts toward automated bot detection on Twitter relied on hand-crafted feature sets extracted from account metadata and content. Varol et al. (2017) proposed BotOrNot (later renamed Botometer), a supervised classification system that aggregated over 1,000 features spanning six categories — user profile, friends, network, temporal, content, and sentiment — and trained a Random Forest classifier to produce a bot probability score [1]. The system demonstrated strong performance on contemporary bot datasets, achieving AUC scores exceeding 0.95 on the Caverlee 2011 dataset. However, subsequent evaluations by Rauchfleisch and Kaiser (2020) revealed significant degradation in accuracy when applied to non-English accounts and accounts from regions with different usage patterns, highlighting the feature set's cultural and linguistic bias [2].

Davis et al. (2016) focused on temporal activity patterns as a discriminative signal, constructing time-series features from tweet inter-arrival times and analysing periodicity through frequency-domain methods [3]. Their approach was effective against first-generation bots that operated on fixed schedules (e.g., one tweet every 15 minutes), but failed to detect bots employing stochastic scheduling with human-like inter-arrival time distributions.

Cresci et al. (2017) introduced the concept of "social spambots" — a new generation of bots that had evolved specifically to evade feature-based classifiers [4]. Through a large-scale empirical study, they demonstrated that these bots maintained follower-to-friend ratios, posting frequencies, and profile completeness metrics that were statistically indistinguishable from genuine accounts when analysed feature-by-feature. Their work provided strong empirical evidence that profile-level features alone were approaching a ceiling in discriminative power.

Yang et al. (2019) proposed a scalable detection framework based on a smaller set of 12 features selected for robustness against manipulation, including account age normalised by tweet count, the proportion of tweets containing URLs, and the fraction of followers who are themselves verified [5]. While computationally efficient, their system shared the fundamental limitation of all feature-only approaches: it could not exploit relational patterns between accounts.

The DNA-inspired approach of Cresci et al. (2016) represented a notable departure from conventional feature engineering [6]. They encoded each account's tweet sequence as a string of characters representing different tweet types (original tweet, reply, retweet, quote) and applied sequence alignment algorithms borrowed from bioinformatics to detect coordinated bot campaigns. While this method captured some behavioural patterns beyond static features, it still operated on individual accounts without considering graph structure.

**Table 2.1: Summary of Traditional Bot Detection Approaches**

| Author(s) | Year | Method | Features Used | Reported Accuracy | Key Limitation |
|-----------|------|--------|---------------|-------------------|----------------|
| Varol et al. [1] | 2017 | Random Forest | 1,000+ multi-category | AUC > 0.95 | Language/cultural bias |
| Davis et al. [3] | 2016 | Temporal analysis | Inter-arrival times | ~91% | Fails against stochastic bots |
| Cresci et al. [4] | 2017 | Ensemble classifier | Profile + content | ~87% | Social spambots evade detection |
| Yang et al. [5] | 2019 | Gradient Boosting | 12 robust features | ~86% | No relational signals |
| Cresci et al. [6] | 2016 | DNA sequencing | Tweet-type sequences | ~93% | Individual accounts only |

## **2.2 GRAPH-BASED BOT DETECTION METHODS**

The recognition that isolated feature-based methods were reaching a performance ceiling prompted a shift toward graph-based approaches. The central idea is straightforward: by modelling the relationships *between* accounts (follows, mentions, retweets), detection systems can access discriminative signals that are invisible at the level of individual profiles.

Fazil and Abulaish (2018) constructed a follow-graph of Twitter accounts and applied community detection algorithms to identify clusters of mutually-following bot accounts [7]. Their approach revealed that bot accounts often form dense, identifiable subgraphs within the broader network. However, their method required the full follower graph to be available, which is computationally expensive to obtain and does not scale to real-time analysis of individual accounts.

Feng et al. (2021) proposed TwiBot-20, a benchmark dataset that included follow-relationship edges alongside user features and tweet content [8]. They demonstrated that graph-aware models consistently outperformed graph-unaware baselines on their benchmark. Their work also highlighted the importance of standardised benchmarks for fair comparison of bot detection systems.

Alhosseini et al. (2019) applied Graph Convolutional Networks (GCNs) to bot detection, constructing a follow-graph and learning node representations that incorporated neighbourhood information [9]. Their GCN-based detector showed improvements of 4–6% in accuracy over feature-only baselines. However, their graph construction used only a single edge type (follow), discarding the rich relational information available from mentions, replies, and content co-occurrence.

Ali Alhosseini et al. (2019) extended this work by incorporating both follower and mention edges, but still treated all edges as belonging to a single relation type — a fundamental limitation that motivated the development of multi-relational approaches [9].

## **2.3 GRAPH NEURAL NETWORKS FOR SOCIAL MEDIA ANALYSIS**

Graph Neural Networks (GNNs) generalise traditional neural network architectures from grid-structured data (images, sequences) to graph-structured data. The fundamental operation in any GNN is *message passing*: each node aggregates information from its neighbours, transforms the aggregated message, and updates its own representation. Multiple rounds of message passing allow each node to incorporate information from increasingly distant parts of the graph.

Kipf and Welling (2017) introduced the Graph Convolutional Network (GCN), which approximates spectral graph convolutions with a first-order Chebyshev polynomial expansion [10]. The layer-wise propagation rule for a GCN is:

**H**^(l+1) = σ( **D̃**^(-1/2) **Ã** **D̃**^(-1/2) **H**^(l) **W**^(l) )

where **Ã** = **A** + **I** is the adjacency matrix with self-loops, **D̃** is the corresponding degree matrix, **H**^(l) is the matrix of node representations at layer *l*, **W**^(l) is the trainable weight matrix, and σ is a non-linear activation function. The key insight is that this formulation implicitly performs feature smoothing across the graph — each node's new representation is a weighted average of its own features and its neighbours' features, transformed by a shared weight matrix.

Veličković et al. (2018) proposed the Graph Attention Network (GAT), which replaces the fixed normalisation coefficients in GCN with learned attention weights [11]. Each node attends to its neighbours with different weights, allowing the model to assign higher importance to more informative neighbours. While GAT has demonstrated strong performance on node classification benchmarks, its attention mechanism operates on a single edge type; extending it to multi-relational graphs requires additional architectural modifications.

Hamilton et al. (2017) introduced GraphSAGE, a framework that learns node representations by sampling and aggregating features from a node's local neighbourhood [12]. Unlike GCN, which requires the full graph adjacency matrix during training, GraphSAGE operates on sampled subgraphs and is therefore more scalable to large graphs. GraphSAGE supports several aggregation functions (mean, LSTM, pooling), providing flexibility in how neighbourhood information is combined.

A common limitation of GCN, GAT, and GraphSAGE in the context of social media bot detection is their treatment of all edges as belonging to a single, homogeneous relation type. In a Twitter social graph, a follower edge and a mention edge carry fundamentally different semantic meanings — a user following another user indicates social interest, while a mention indicates direct interaction. Treating these relations identically discards information that may be critical for accurate classification.

## **2.4 RELATIONAL GRAPH CONVOLUTIONAL NETWORKS**

Schlichtkrull et al. (2018) introduced the Relational Graph Convolutional Network (RGCN), extending the GCN framework to handle multi-relational data [13]. The core modification is the use of *relation-specific* weight matrices: instead of a single shared weight matrix **W** for all edges, the RGCN maintains a separate weight matrix **W**_r for each relation type *r*. The propagation rule becomes:

**h**_i^(l+1) = σ( **W**_0^(l) **h**_i^(l) + Σ_{r∈R} Σ_{j∈N_i^r} (1 / c_{i,r}) **W**_r^(l) **h**_j^(l) )

where **h**_i^(l) is the representation of node *i* at layer *l*, N_i^r is the set of neighbours of node *i* connected via relation type *r*, c_{i,r} is a normalisation constant (typically |N_i^r|), **W**_r^(l) is the weight matrix specific to relation *r*, and **W**_0^(l) is the self-loop weight matrix. The outer summation over R allows the model to aggregate information from different relation types with different learned transformations.

This formulation has direct applicability to the bot detection problem: follower relationships, mention patterns, reply chains, and content co-occurrence each represent distinct relation types that carry different discriminative signals. An RGCN can learn, for example, that being followed by accounts with default profile images (relation: follower) is a stronger indicator of bot status than sharing a common hashtag with such accounts (relation: hashtag co-occurrence).

To mitigate the parameter explosion that arises from maintaining |R| separate weight matrices (each of dimension d^(l) × d^(l+1)), Schlichtkrull et al. proposed basis decomposition:

**W**_r^(l) = Σ_{b=1}^{B} a_{rb}^(l) **V**_b^(l)

where **V**_b^(l) are shared basis matrices and a_{rb}^(l) are relation-specific scalar coefficients. This reduces the number of parameters from O(|R| × d^2) to O(B × d^2 + |R| × B), which is significantly smaller when B << |R|.

**Table 2.2: Comparison of GNN Architectures for Bot Detection**

| Architecture | Multi-Relational | Attention | Neighbourhood Sampling | Parameter Complexity |
|-------------|-----------------|-----------|----------------------|---------------------|
| GCN [10] | No | No | No (full graph) | O(d²) per layer |
| GAT [11] | No | Yes | No (full graph) | O(d² + d) per layer |
| GraphSAGE [12] | No | Optional | Yes | O(d²) per layer |
| RGCN [13] | Yes | No | No (full graph) | O(R × d²) per layer |

## **2.5 THE MGTAB BENCHMARK**

Shi et al. (2023) proposed MGTAB (Multi-Relational Graph-Based Twitter Account Detection Benchmark), a purpose-built benchmark for evaluating graph-based bot detection methods [14]. The benchmark comprises:

- **10,199 Twitter accounts** with expert-verified annotations (human = 0, bot = 1).
- **Seven relation types**: follower, friend, mention, reply, quote, URL co-occurrence, and hashtag co-occurrence.
- **788-dimensional node features**: 20 normalised profile features concatenated with 768-dimensional LaBSE embeddings of each user's tweet history.
- **Pre-computed graph structure**: edge indices, edge types, and edge weights stored as PyTorch tensors.

The authors evaluated multiple GNN architectures on this benchmark, including GCN, GAT, GraphSAGE, and RGCN. Their results demonstrated that multi-relational models (specifically RGCN) consistently outperformed single-relation baselines, confirming the hypothesis that the distinction between relation types carries significant discriminative information.

A critical contribution of the MGTAB work is the standardisation of the feature engineering pipeline — specifically, the use of summed (not averaged, not L2-normalised) LaBSE pooler outputs for the tweet embedding dimensions. This detail, which is not prominently documented in the original paper, proved essential during the reproduction phase of our project; incorrect normalisation of the LaBSE embeddings led to a dramatic drop in model accuracy during live inference, as detailed in Chapter 4.

## **2.6 SUMMARY OF LITERATURE REVIEW**

The review of existing literature reveals a clear progression from feature-based to graph-based bot detection methods, driven by the increasing sophistication of automated accounts. Feature-only approaches (BotOrNot, DNA-based sequencing, temporal analysis) have demonstrated declining effectiveness against modern social spambots. The introduction of graph-based methods, particularly graph neural networks, provided access to relational and structural signals that are harder for bot operators to fabricate. The RGCN architecture, by maintaining relation-specific weight matrices, is particularly well-suited to the Twitter bot detection task, where different edge types (follower, mention, reply, etc.) carry qualitatively different semantic information.

The MGTAB benchmark provides a standardised evaluation framework for this problem, and the present project adopts it as the foundation for both model training and live inference.

&nbsp;

---

**References for Chapter 2:**

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
