# Collaborative Intrusion Detection Framework

A collaborative multi-model Intrusion Detection System (IDS) integrating Random Forest (RF), Deep Neural Network (DNN) and Long Short-Term Memory (LSTM) within a unified detection framework.

---

## Implementation Pipeline

### 1. Data Acquisition & Processing
- Column standardisation
- Dataset merging
- Data cleaning
- Data partitioning
- Data balancing

### 2. Base ML/DL Model Implementation
- Feature encoding & scaling
- Independent model training:
  - Random Forest (RF)
  - Deep Neural Network (DNN)
  - Long Short-Term Memory (LSTM)

### 3. Collaborative Detection Approach
- Collection of individual model predictions
- Implementation of collaborative framework:
  - Majority Voting strategy
  - Weighted Voting strategy
- Output probability normalisation
- Final shared decision

### 4. Evaluation & Comparison
- Binary intrusion detection evaluation
- Multi-class attack classification evaluation
- False Positive Rate (FPR) / False Negative Rate (FNR) analysis
- Collaborative framework analysis
- Performance evaluation

---

## Dataset Information
This project was trained and evaluated using the following datasets:
- CIC-UNSW-NB15
  - https://www.unb.ca/cic/datasets/cic-unsw-nb15.html
- CIC-IDS-2017
  - https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS-2018
  - https://www.unb.ca/cic/datasets/ids-2018.html
- CIC-DDoS-2019
  - https://www.unb.ca/cic/datasets/ddos-2019.html

The processed and balanced intrusion detection dataset used during training, validation and testing can be accessed using the following link:
- https://www.kaggle.com/datasets/merluke/intrusion-detection-dataset
 
---

## Project Structure

```text
collaborative_ids/
│
├── data/
│   ├── cleaned/
│   │   └── partitioned/
│   │       └── # Contains partitioned datasets
│   │
│   └── original/
│       └── # Contains original datasets
│
├── models/
│   ├── dnn/
│   ├── lstm/
│   └── rf/
│       └── # Each model directory can contain:
│           ├── final_model.joblib / final_model.keras
│           ├── encoder.joblib
│           ├── features.joblib
│           ├── scaler.joblib
│           └── pca.joblib
│
├── predictions/
│   └── # Contains individual model predictions and
│       # aggregated collaborative predictions
│
├── scripts/
│   ├── model_server.py
│   ├── run_model.py
│   └── voting_system.py
│
├── 01_data_processing.ipynb
├── 02_base_model_training.ipynb
├── 03_voting_mechanism_analysis.ipynb
└── 04_realtime_traffic_evaluation.ipynb
```

---

## Running the Collaborative Framework

### 1. Convert Network Traffic to Flows

Before running the framework, network traffic must be converted into flow-based records using the CICFlowMeter tool.

Since the RF, DNN and LSTM models were trained on flow-based network traffic data, the framework requires input in the same format. Therefore, raw packets must first be processed by CICFlowMeter to generate the aggregated flow features expected by the models.

### 2. Start the Model Server
Run:
```bash
python scripts/model_server.py
```

This generates predictions from the individual RF, DNN and LSTM models.
Model files must be stored in:
 - models/<model_name>/

### 3. Run the Collaborative Voting Framework
Run:
```bash
python scripts/voting_system.py
```

This aggregates individual model predictions and produces the final collaborative decision using:
- Majority Voting
- Weighted Voting
