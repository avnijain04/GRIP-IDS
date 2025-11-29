# **GRIP-IDS**

### **Bi-LSTM/CNN Hierarchical Explainable Federated IDS for Secure Critical IoT**

This repository contains the full implementation of **GRIP-IDS**, an Intrusion Detection System designed for critical IoT infrastructure.
The project integrates:

* Hybrid **CNN + Bi-LSTM** modeling
* **Federated Learning** (FedAvg, FedProx, FedNova)
* **Robust Aggregation** (Multi-Krum, poisoning attack simulation)
* **Secure Aggregation** using AES-GCM and Ed25519 signatures
* **Explainability** through SHAP

This work was developed as a **minor project** and is accompanied by a **published research paper** based on this implementation.

---

# **1. Team Members**

**Project Authors:**

* **Avni Jain**
* **Akhilesh Chouhan**
* **Ashish Tiwari**
* **Aman Khan**

---

# **2. Project Overview**

Modern critical IoT networks face three core challenges:

1. **Data privacy** — Raw traffic cannot be centrally collected
2. **Attack resilience** — Federated learning is vulnerable to malicious clients
3. **Explainability** — Security analysts require interpretable outputs

**GRIP-IDS** addresses these challenges with a layered architecture:

1. A **hybrid CNN–BiLSTM** model to learn temporal–spatial intrusion patterns
2. **Federated IDS** to maintain data locality
3. **Robust FL** to defend against poisoning attacks
4. **Secure aggregation** to prevent tampering
5. **Explainable AI** using SHAP to highlight influential features

---

# **3. Features**

### **Hybrid Deep Learning**

* CNN for spatial feature extraction
* Bi-LSTM for temporal patterns
* Trained baseline models: CNN, LSTM, Hybrid

### **Federated Learning**

* FedAvg
* FedProx
* FedNova
* Non-IID data partitioning
* Per-round metrics and logs

### **Robust FL**

* Multi-Krum aggregation
* Label-flip poisoning attack simulation
* Round-wise confusion matrices

### **Secure Federated Learning**

* AES-GCM encrypted client updates
* Ed25519 digital signatures
* Verification of every client payload
* Detection of malformed or tampered updates

### **Explainability**

* SHAP KernelExplainer
* Beeswarm and feature-importance plots
* Top contributing features JSON export

---

# **4. Directory Structure**

```
├── data/
│   ├── train_test_network.csv
│   └── processed/
│       ├── X_train.csv / X_train.npy
│       ├── X_test.csv  / X_test.npy
│       ├── y_train.csv / y_train.npy
│       └── y_test.csv  / y_test.npy
│
├── models/
│   ├── cnn, lstm, hybrid baselines
│   ├── fedavg / fedprox / fednova
│   ├── multikrum robust model
│   └── secure_global.h5
│
├── results/
│   ├── training metrics & plots
│   ├── federated logs & curves
│   ├── SHAP explainability
│   ├── attack evaluations
│   └── secure FL timing comparison
│
└── src/
    ├── preprocess.py
    ├── inspect_data.py
    ├── train.py
    ├── shap_explain.py
    ├── federated_sim.py
    ├── federated_robust.py
    ├── federated_secure.py
    ├── model_defs.py
    ├── fl_utils.py
    ├── crypto_utils.py
    ├── multi_krum.py
    ├── plot_metrics.py
    ├── model_footprint.py
    └── config.py
```

---

# **5. Setup Instructions**

### **Create and activate environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.\.venv\Scripts\activate         # Windows
```

### **Install dependencies**

```bash
pip install -r requirements.txt
```

### **Run preprocessing**

```bash
python src/preprocess.py
```

---

# **6. Running the Pipeline**

### Baseline Models

```bash
python src/train.py
```

### Explainability

```bash
python src/shap_explain.py
```

### Federated Learning

```bash
python src/federated_sim.py
```

### Robust FL

```bash
python src/federated_robust.py
```

### Secure FL

```bash
python src/federated_secure.py
```

### Model Footprint

```bash
python src/model_footprint.py
```

### Full automated pipeline (PowerShell)

```powershell
.\run_all.ps1
```

---

# **7. Outputs**

After running the pipeline, the `results/` directory includes:

* Accuracy & loss curves
* Confusion matrices
* ROC & PR curves
* Federated logs (round-wise)
* Robust FL evaluations
* Secure aggregation timing plots
* SHAP feature-importance and beeswarm plots
* Model footprint summary

---

# **8. Research Paper**

This repository accompanies the research work conducted for the minor project.
The paper documents:

* Model architecture
* Federated and robust extensions
* Secure aggregation mechanism
* Explainability results
* Evaluation across IoT traffic dataset

(Citation details can be added here once the paper is formally published.)

---

# **9. License**

This project is intended for academic and research use.
You may modify or extend it with proper attribution.

---

# **10. Contact**

For questions or replication assistance:

**Team:**

* Avni Jain
* Akhilesh Chouhan
* Ashish Tiwari
* Aman Khan

