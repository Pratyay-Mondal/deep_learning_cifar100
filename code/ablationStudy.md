# Task 2: Ablation Study
**Target:** Optimize for CIFAR-100 dataset (>50% Accuracy)

| Exp ID | Change Description | Hyperparameters | Train Acc | Test Acc | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *2.0* | *Baseline Switch to CIFAR-100* | Epochs=5, No Augmentation | 76.19% | 46.12% | **Result:** FAILED (<50%). <br>**Issue:** Severe overfitting (30% gap between Train/Val). Model memorized training data. <br>**Fix:** Needs data augmentation to prevent memorization and more epochs to converge. |
| *2.1* | *Data Augmentation and Epochs* | Epochs=20, RandomCrop/RandomFlip/RandomRotation | 55.43% | **55.21%** | **Result:** PASSED (>50%). <br>**Success:** Overfitting cured (Train $\approx$ Test). Augmentation successfully bridged the generalization gap. <br>**Decision:** Selected as final model due to superior performance and simplicity. |
