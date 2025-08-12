# UCL x Lifemote – Predictive Analytics on WiFi Data

This project applies predictive analytics and machine learning to proactively manage customer experience and reduce operational costs for Internet Service Providers (ISPs). Developed in collaboration with **Lifemote**, a Turkish SaaS company specializing in broadband network analytics, the work focuses on two key objectives:
1. **Predict broadband connectivity issues** within the next 7 days using 14 days of historical network data. (Succeeded)
2. **Identify customers likely to contact customer service** based on 3 days of recent network behavior. (Failed)

---

## Project Overview

**Goal:** Enable ISPs to take proactive actions by predicting future faults and anticipating customer calls, reducing churn, improving service quality, and optimizing operational resources.

**Key Benefits for ISPs:**
- Reduce inventory holding costs by forecasting the number of faults.
- Optimize workforce scheduling.
- Increase customer satisfaction by resolving issues before customers notice.

---

## Repository Structure

- **Notebooks**
  - [Predicting WiFi Connectivity Issues](https://colab.research.google.com/drive/1U8C83hP4goEIJdYdnpwQ7ant9txX7mW_?usp=sharing)
  - [Predicting Customer Calls using Autoencoder (Unsupervised)](https://colab.research.google.com/drive/1hOh0yimT-2cBj3Bw2hUYsouoK5HjRtZl?usp=sharing)
  - [Predicting Customer Calls using IsolationForest + Classification (Semi-supervised)](https://colab.research.google.com/drive/1sBOgsxIEa6uQw9_Spo0IWgp-in4ZkIKS?usp=sharing)
  - [Cluster Analysis of Customer Calls](https://colab.research.google.com/drive/1yXpHQf76k3E5xdmFsaCdYjwV9AvldTo2?usp=sharing)

- **Data**: Network and call log datasets (not included in repository due to confidentiality).

---

## Methodology

### 1. Data Preparation
- **Data Sources**: 30 days of network behavior metrics & customer call logs.
- **Preprocessing**:
  - Remove low-variance and high-missing-rate features (> 50% missing rate).  
  - Merge network and call datasets on device_id (network data), resource_id (call data), and analysis_date
  - Create the target label flag_fault (connectivity issue) from wifi_qoe_status. If wifi_qoe_status is 'poor', label it as a connectivity issue (1); otherwise, label it as      no connectivity issue (0).
  - Create the target label flag_call (customer call). If any resource_id matches a device_id in the network data, label it as a caller (1); otherwise, label it as a non-        caller (0).

### 2. Feature Engineering
- **TSFRESH** for time-series feature extraction.
- **Rolling Windows** Fault prediction (features 14 days, label 7 days), Call Prediction (features 3 days, label 1 day)
- Manual features for call prediction (lag, delta, ratios).
- Multicollinearity check via Variance Inflation Factor (VIF). (> 10 were dropped)

### 3. Modeling
#### **Connectivity Issues Prediction**
- Models: Random Forest, XGBoost, CatBoost, LightGBM.
- Evaluation Metrics: Precision, Recall, F1, ROC AUC, PR AUC.

#### **Customer Call Prediction**
- **Semi-supervised**: IsolationForest anomaly scores + classification.
- **Unsupervised**: Autoencoder anomaly detection.
- Evaluation Metrics: PR AUC, Precision@K, Recall@K.

#### **Clustering**
- DBSCAN on UMAP-projected data to detect caller behavior patterns.

---

## Key Results

**Connectivity Issues Prediction (Best Model: XGBoost)**
| Metric   | Score  |
|----------|--------|
| Precision| 0.6445 |
| Recall   | 0.7617 |
| F1-score | 0.6982 |
| PR-AUC   | 0.7742 |
| ROC-AUC  | 0.8487 |

**Customer Call Prediction**
- Semi-supervised & unsupervised methods performed poorly due to extreme class imbalance (0.07% positive).
- Abnormal network behavior alone is not a strong predictor of customer calls.

**DBSCAN Clustering Analysis**
- 8 distinct caller behavior patterns identified, offering insights for targeted maintenance.

---

## Usage

Open the desired Colab notebook from the links above and run the cells in order.

---

## Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost, TSFRESH, SHAP, Optuna, Matplotlib, Seaborn.
- **ML Techniques**: Supervised learning, semi-supervised learning, anomaly detection, clustering.
