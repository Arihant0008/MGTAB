
# FAKE PROFILE DETECTION USING MGTAB ON TWITTER

### A Thesis Submitted in Partial Fulfilment of the Requirements for the Degree of Bachelor of Engineering in Computer Science and Engineering

**Shri Ramdeobaba College of Engineering and Management, Nagpur**
**(An Autonomous Institute Affiliated to Rashtrasant Tukadoji Maharaj Nagpur University)**

Academic Year 2025–2026

---

---

<center><b>ACKNOWLEDGEMENTS</b></center>

&nbsp;

We wish to express our sincere gratitude to our project guide for their sustained guidance, constructive criticism, and encouragement throughout the course of this project work. Their willingness to devote time to regular discussions and reviews has been instrumental in shaping the direction and rigour of this investigation.

We are thankful to the Head of the Department of Computer Science and Engineering, Shri Ramdeobaba College of Engineering and Management, Nagpur, for providing the necessary facilities and an academic environment that is conducive to research and development activities.

We also extend our gratitude to the faculty members of the Computer Science and Engineering department who, through their coursework and informal discussions, have equipped us with the theoretical foundations upon which this project rests — particularly in the areas of machine learning, graph theory, and web application engineering.

We acknowledge the contribution of the open-source community, specifically the developers of PyTorch, PyTorch Geometric, the Hugging Face Transformers library, FastAPI, and React, whose freely available tools and documentation made the technical implementation of this project feasible within an academic timeline.

Finally, we wish to thank our families and peers for their patience and moral support during the execution of this project.

&nbsp;

---

<center><b>ABSTRACT</b></center>

&nbsp;

The proliferation of automated accounts — commonly referred to as bots — on the Twitter/X platform presents a persistent threat to public discourse, electoral integrity, and the credibility of online information ecosystems. Traditional detection approaches that rely on metadata features extracted from individual user profiles (such as follower count, account age, and tweet frequency) have demonstrated diminishing effectiveness against modern bots that employ large language models for content generation and mimic organic behavioural patterns. The fundamental limitation of these methods is their treatment of each account as an isolated data point, ignoring the structural patterns embedded in social network topology.

This project addresses the stated limitation by implementing a full-stack bot detection system grounded in the MGTAB (Multi-Relational Graph-Based Twitter Account Detection Benchmark) framework. The core detection mechanism employs a Relational Graph Convolutional Network (RGCN), a specialised variant of graph neural networks that operates on heterogeneous multi-relational graphs. Unlike standard Graph Convolutional Networks that assume a single, uniform edge type, the RGCN maintains separate learnable weight matrices for each relation type, thereby preserving the semantic distinction between follower, friend, mention, reply, quote, URL co-occurrence, and hashtag co-occurrence relationships.

The system constructs a 788-dimensional node feature vector for each Twitter account, composed of 20 normalised profile metadata features and a 768-dimensional tweet embedding produced by the Language-Agnostic BERT Sentence Embedding (LaBSE) model. These node features, together with a multi-relational edge structure containing seven distinct relation types, are fed into a two-layer RGCN architecture (788 → 256 → 2) that classifies the target node as either human or bot. The model was trained on the MGTAB benchmark dataset comprising 10,199 expert-annotated Twitter accounts, achieving a test accuracy of 88.23% and a bot recall of 90.29%, outperforming baseline GCN, GAT, and GraphSAGE models.

The production system is implemented as a web application with a React 19 frontend and a FastAPI backend. A real-time data ingestion pipeline, powered by the Scweet library, scrapes the target user's profile, recent tweets, and ego-graph neighbourhood from Twitter's internal GraphQL API using cookie-based authentication. The backend streams progress updates to the frontend via Server-Sent Events (SSE), mitigating timeout issues during the approximately 90-second scraping process. The application is deployed live at https://www.mgtab.me/, with the backend hosted on Hugging Face Spaces (Docker) and the frontend served through Vercel CDN.

This thesis details the mathematical formulation of the RGCN message-passing scheme, the system architecture and data flow pipeline, the feature engineering methodology, and a comparative evaluation of graph neural network architectures on the MGTAB benchmark.

&nbsp;

---

<center><b>TABLE OF CONTENTS</b></center>

&nbsp;

| Sr. No. | Title | Page No. |
|---------|-------|----------|
| | Acknowledgements | i |
| | Abstract | ii |
| | Table of Contents | iii |
| | List of Figures | v |
| | List of Tables | vi |
| | List of Symbols, Abbreviations and Nomenclature | vii |
| **1** | **INTRODUCTION** | **1** |
| 1.1 | General | 1 |
| 1.2 | Problem Statement | 2 |
| 1.3 | Aim and Objectives | 3 |
| 1.4 | Scope of the Project | 4 |
| 1.5 | Motivation for Multi-Graph Approaches | 5 |
| 1.6 | Organisation of the Thesis | 6 |
| **2** | **REVIEW OF LITERATURE** | **8** |
| 2.1 | Feature-Based Bot Detection Methods | 8 |
| 2.2 | Graph-Based Bot Detection Methods | 10 |
| 2.3 | Graph Neural Networks for Social Media Analysis | 12 |
| 2.4 | Relational Graph Convolutional Networks | 14 |
| 2.5 | The MGTAB Benchmark | 16 |
| 2.6 | Summary of Literature Review | 17 |
| **3** | **SYSTEM ARCHITECTURE AND DESIGN** | **19** |
| 3.1 | High-Level Architecture | 19 |
| 3.2 | Frontend Architecture | 21 |
| 3.2.1 | React Component Hierarchy | 21 |
| 3.2.2 | Client-Side State Management | 22 |
| 3.2.3 | SSE Stream Consumer | 23 |
| 3.3 | Backend Architecture | 24 |
| 3.3.1 | FastAPI Application Structure | 24 |
| 3.3.2 | API Route Design | 25 |
| 3.3.3 | Asynchronous Task Handling and SSE | 27 |
| 3.4 | Deployment Infrastructure | 28 |
| **4** | **DATA INGESTION AND FEATURE ENGINEERING** | **30** |
| 4.1 | Twitter Data Acquisition Pipeline | 30 |
| 4.1.1 | Cookie-Based Authentication | 30 |
| 4.1.2 | Ego-Graph Scraping Strategy | 31 |
| 4.1.3 | Rate-Limit Resilience | 33 |
| 4.2 | Feature Vector Construction | 34 |
| 4.2.1 | Profile Feature Extraction (20-D) | 34 |
| 4.2.2 | Normalization Pipeline | 36 |
| 4.2.3 | LaBSE Tweet Embedding (768-D) | 37 |
| 4.2.4 | Feature Concatenation | 39 |
| 4.3 | Multi-Relational Graph Construction | 39 |
| 4.3.1 | Relation Type Definitions | 39 |
| 4.3.2 | Edge Direction Semantics | 40 |
| 4.3.3 | Graph Filtering and Self-Loop Fallback | 41 |
| **5** | **RGCN MODEL: MATHEMATICAL FOUNDATIONS AND IMPLEMENTATION** | **43** |
| 5.1 | Graph Convolutional Networks — Background | 43 |
| 5.2 | Relational Graph Convolutional Networks | 45 |
| 5.2.1 | Message-Passing Formulation | 45 |
| 5.2.2 | Basis Decomposition | 47 |
| 5.2.3 | Layer-wise Propagation Rule | 48 |
| 5.3 | Model Architecture | 49 |
| 5.4 | Training Procedure | 50 |
| 5.4.1 | Loss Function and Class Imbalance Correction | 50 |
| 5.4.2 | Optimiser and Hyperparameters | 51 |
| 5.4.3 | Dataset Splits | 52 |
| 5.5 | Inference Pipeline | 52 |
| **6** | **RESULTS AND DISCUSSIONS** | **54** |
| 6.1 | Experimental Setup | 54 |
| 6.2 | Comparative Performance of GNN Architectures | 55 |
| 6.3 | Accuracy and Loss Curves | 57 |
| 6.4 | Confusion Matrix Analysis | 58 |
| 6.5 | Precision, Recall, and F1-Score | 59 |
| 6.6 | Feature Importance Analysis | 60 |
| 6.7 | Live Deployment Validation | 61 |
| 6.8 | Discussion | 62 |
| **7** | **SUMMARY AND CONCLUSIONS** | **64** |
| 7.1 | Summary | 64 |
| 7.2 | Conclusions | 65 |
| 7.3 | Scope for Further Work | 66 |
| | List of Publications | 68 |
| | References | 69 |
| | Appendix A: API Schema Definitions | 72 |
| | Appendix B: Selected Code Listings | 74 |
| | Appendix C: Data Dictionary | 76 |

&nbsp;

---

<center><b>LIST OF FIGURES</b></center>

&nbsp;

| Figure No. | Title | Page No. |
|------------|-------|----------|
| 1.1 | Evolution of Twitter Bot Sophistication | 3 |
| 1.2 | Isolated vs. Graph-Based Account Analysis | 5 |
| 3.1 | High-Level System Architecture | 20 |
| 3.2 | React Component Hierarchy Diagram | 22 |
| 3.3 | SSE Data Flow Sequence Diagram | 23 |
| 3.4 | FastAPI Route Structure | 25 |
| 3.5 | Deployment Architecture (Vercel + Hugging Face Spaces) | 29 |
| 4.1 | Ego-Graph Scraping Pipeline Flowchart | 32 |
| 4.2 | Rate-Limit Fallback Decision Flow | 33 |
| 4.3 | 788-Dimensional Feature Vector Layout | 35 |
| 4.4 | Log-MinMax Normalization Pipeline | 36 |
| 4.5 | LaBSE Encoding and Summation Pipeline | 38 |
| 4.6 | Seven Relation Types in the MGTAB Graph | 40 |
| 4.7 | Example Mini Ego-Graph for Inference | 41 |
| 5.1 | Standard GCN Message-Passing (Single Relation) | 44 |
| 5.2 | RGCN Message-Passing with Relation-Specific Weights | 46 |
| 5.3 | RGCN Model Architecture (788 → 256 → 2) | 49 |
| 5.4 | Training Pipeline Flowchart | 51 |
| 5.5 | Inference Pipeline from Request to Prediction | 53 |
| 6.1 | Comparative Test Accuracy Across GNN Models | 56 |
| 6.2 | Training Loss Curve over 200 Epochs | 57 |
| 6.3 | Training and Validation Accuracy Curves | 57 |
| 6.4 | Confusion Matrix — RGCN on Test Set | 58 |
| 6.5 | Bot Recall Comparison Across Models | 59 |
| 6.6 | Top-5 Features by Information Gain | 61 |
| 6.7 | Screenshot — Live Detection at mgtab.me | 62 |

&nbsp;

---

<center><b>LIST OF TABLES</b></center>

&nbsp;

| Table No. | Title | Page No. |
|-----------|-------|----------|
| 2.1 | Summary of Traditional Bot Detection Approaches | 11 |
| 2.2 | Comparison of GNN Architectures for Bot Detection | 15 |
| 3.1 | FastAPI Endpoint Summary | 26 |
| 3.2 | Frontend Route Definitions | 22 |
| 4.1 | Scweet Scraper Configuration Parameters | 31 |
| 4.2 | Profile Feature Definitions and Index Mapping | 35 |
| 4.3 | MinMax Normalization Bounds from MGTAB Dataset | 37 |
| 4.4 | Seven MGTAB Relation Types with Direction Semantics | 40 |
| 5.1 | RGCN Hyperparameter Summary | 51 |
| 5.2 | MGTAB Dataset Split Statistics | 52 |
| 6.1 | Comparative Results — GNN Architectures on MGTAB | 55 |
| 6.2 | RGCN Classification Report (Precision, Recall, F1) | 59 |
| 6.3 | Feature Importance Ranking | 60 |
| 6.4 | Pipeline Timing Breakdown | 62 |

&nbsp;

---

<center><b>LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURE</b></center>

&nbsp;

| Symbol / Abbreviation | Description |
|-----------------------|-------------|
| RGCN | Relational Graph Convolutional Network |
| GCN | Graph Convolutional Network |
| GAT | Graph Attention Network |
| GNN | Graph Neural Network |
| MGTAB | Multi-Relational Graph-Based Twitter Account Detection Benchmark |
| LaBSE | Language-Agnostic BERT Sentence Embedding |
| BERT | Bidirectional Encoder Representations from Transformers |
| SSE | Server-Sent Events |
| API | Application Programming Interface |
| REST | Representational State Transfer |
| PyG | PyTorch Geometric |
| CDN | Content Delivery Network |
| CORS | Cross-Origin Resource Sharing |
| JSON | JavaScript Object Notation |
| DOM | Document Object Model |
| JSX | JavaScript XML |
| CSV | Comma-Separated Values |
| HTTP | HyperText Transfer Protocol |
| URL | Uniform Resource Locator |
| **x** | Node feature matrix, x ∈ ℝ^(N × d) |
| **W**_r | Relation-specific weight matrix for relation r |
| **W**_0 | Self-loop weight matrix |
| **h**_i^(l) | Hidden representation of node i at layer l |
| N_i^r | Set of neighbours of node i under relation r |
| c_{i,r} | Normalisation constant, typically \|N_i^r\| |
| σ | Non-linear activation function (ReLU) |
| d | Input feature dimensionality (788) |
| d_h | Hidden layer dimensionality (256) |
| K | Number of output classes (2) |
| R | Number of relation types (7) |
| N | Number of nodes in the graph |
| E | Number of edges in the graph |
| F1 | Harmonic mean of precision and recall |
| TP | True Positive |
| FP | False Positive |
| FN | False Negative |
| TN | True Negative |

&nbsp;

---
