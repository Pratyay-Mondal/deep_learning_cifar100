# 🔍 Deep Learning Optimization: CIFAR-100 Refactoring

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Stable-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project involves the forensic debugging of a legacy deep learning codebase and the engineering of a robust Convolutional Neural Network (CNN) pipeline.

The goal was twofold:
1.  **Stabilization (Task 1):** Audit a broken codebase to identify and fix critical architectural bugs preventing convergence on CIFAR-10.
2.  **Optimization (Task 2):** Adapt the stabilized model to the harder **CIFAR-100** dataset, achieving >50% accuracy through data-centric optimization strategies (Data Augmentation) rather than just increasing model complexity.

---

## 🛠️ Key Achievements

### Task 1: Forensic Code Audit (Stabilization)
Identified and resolved **9 critical bugs** that were causing gradient explosion, data leakage, and silent failures:
* **Data Pipeline:** Fixed normalization constants (removing "exploding input" bug) and separated Train/Test transforms to prevent data leakage.
* **Architecture:** Corrected tensor dimension mismatches in MLP and CNN definitions.
* **Training Loop:** Implemented missing `optimizer.zero_grad()` to fix infinite gradient accumulation.
* **Loss Function:** Swapped `NLLLoss` for `CrossEntropyLoss` to match model output (Logits vs LogSoftmax).
* **Evaluation:** Fixed a critical bug where the model was evaluated on the *Training Set* instead of the *Test Set*.

### Task 2: CIFAR-100 Optimization
Addressed severe overfitting (Train 76% / Test 46%) caused by data scarcity in CIFAR-100.
* **Strategy:** Implemented a robust **Data Augmentation** pipeline (`RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`).
* **Outcome:** Forced the model to learn invariant features, closing the generalization gap.
* **Final Accuracy:** **55.47%** (Surpassing the 50% target).

---

## 📊 Results & Ablation Study

| Experiment | Change Description | Epochs | Train Acc | Test Acc | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2.0** | Baseline CIFAR-100 | 5 | 76.19% | 46.12% | **FAILED.** Severe overfitting (30% gap). |
| **2.1** | **+ Data Augmentation** | 20 | 55.43% | **55.47%** | **PASSED.** Generalization gap closed. Stable convergence. |

> **Note on Extended Experiments:**
> Beyond the submitted model, I conducted benchmarks using deeper architectures (ResNet, 3-Block CNN) on Google Colab for 100 epochs. These experiments yielded a peak accuracy of **62.74%**, demonstrating further potential for this pipeline.

---

## 🚀 Setup & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Pratyay-Mondal/deep_learning_cifar100.git
cd deep_learning_cifar100/code
pip install -r requirements.txt
```

### 2. Training
To train the model from scratch (CIFAR-100):
```bash
python train.py
```
* **Config:** Modify ```config.py``` to change hyperparameters (Epochs, Batch Size, Learning Rate).



### 3. Evaluation (Reproducibility)
To verify the results using the pre-trained weights without retraining:
```bash
python evaluate_model.py
```



### Expected Output:
```bash
========================================
FINAL RESULT: Task 2 (CIFAR-100)
========================================
Test Accuracy: 55.47%
Test Loss:     1.6249
========================================
```



### 📂 Project Structure
```bash
├── checkpoints/          # Saved model weights
│   └── best_model_cnn.pth
├── config.py             # Hyperparameters & System settings
├── data_loader.py        # Dataset handling & Augmentation pipelines
├── models.py             # CNN & MLP Architecture definitions
├── train.py              # Main training loop
├── utils.py              # Helper functions (training steps, plotting)
├── evaluate_model.py     # Standalone script for result verification
└── README.md             # Project documentation
```



### 🏆 Final Model Checkpoint
* **File:** ```checkpoints/best_model_cnn.pth```
* **Architecture:** Standard CNN (2 Conv Blocks + 2 FC Layers)
* **Input:** 3×32×32
* **Classes:** 100


### 📝 License
This project is for educational purposes as part of the Deep Learning Course.
