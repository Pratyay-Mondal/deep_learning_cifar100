# Task 2: Adapt and Optimize for Production - CIFAR-100 dataset

### The report contains the training, evaluation, testing: before and after oprtimizing the codebase for CIFAR-100 dataset.

### 1st step: 

```text
With using Epochs=5, No Augmentation: 
Achieved Train Accuracy: 76.19%  and Test Accuracy: 46.12%
```

#### Result:
```text
======================================================================
Configuration Settings
======================================================================
BATCH_SIZE.................... 128
DATA_DIR...................... ./data
DEVICE........................ cpu
LEARNING_RATE................. 0.001
MIN_DELTA..................... 0.001
MLP_DROPOUT................... 0.3
MLP_HIDDEN_SIZES.............. [512, 256]
MODEL_TYPE.................... cnn
NUM_CLASSES................... 100
NUM_EPOCHS.................... 5
NUM_WORKERS................... 2
PATIENCE...................... 10
PRINT_EVERY................... 5
RANDOM_SEED................... 42
SAVE_BEST_ONLY................ True
SAVE_DIR...................... ./checkpoints
USE_EARLY_STOPPING............ True
VAL_SPLIT..................... 0.1
WEIGHT_DECAY.................. 0.0005
======================================================================

Using device: cpu

Loading CIFAR-10 data...
Files already downloaded and verified
Files already downloaded and verified
Training samples: 45000
Validation samples: 5000
Test samples: 10000

Creating CNN model...
Total parameters: 2,215,940

======================================================================
Starting Training
======================================================================
Epoch [  1/5] | Train Loss: 2.9655 | Train Acc: 28.08% | Val Loss: 2.5018 | Val Acc: 36.68%
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Epoch [  5/5] | Train Loss: 0.8980 | Train Acc: 76.19% | Val Loss: 2.1441 | Val Acc: 46.02%

======================================================================
Training Completed
======================================================================
Best Validation Accuracy: 46.44%

Generating training history plots...
Training history plot saved to ./checkpoints/training_history_cnn.png
2026-01-09 21:31:25.531 python[76979:16185965] The class 'NSSavePanel' overrides the method identifier.
 This method is implemented by class 'NSWindow'

Evaluating on test set...
Test Loss: 2.1346 | Test Accuracy: 46.12%

======================================================================
All Done!
======================================================================

```

CNN: Training and Validation: Loss and Accuracy
<img width="1400" height="500" alt="task2_CNN_simple" src="https://github.com/user-attachments/assets/12ea1036-3e22-4d2d-a4c4-3e086495fd01" />


### 2nd step: 

```text
With Data Augmentation and increased Epochs
Epochs=20, and RandomCrop/RandomFlip/RandomRotation 
Achieved Train Accuracy: 55.43% and Test Accuracy:  55.21%

```

#### Result:
```text
======================================================================
Configuration Settings
======================================================================
BATCH_SIZE.................... 128
DATA_DIR...................... ./data
DEVICE........................ cpu
LEARNING_RATE................. 0.001
MIN_DELTA..................... 0.001
MLP_DROPOUT................... 0.3
MLP_HIDDEN_SIZES.............. [512, 256]
MODEL_TYPE.................... cnn
NUM_CLASSES................... 100
NUM_EPOCHS.................... 20
NUM_WORKERS................... 2
PATIENCE...................... 10
PRINT_EVERY................... 5
RANDOM_SEED................... 42
SAVE_BEST_ONLY................ True
SAVE_DIR...................... ./checkpoints
USE_EARLY_STOPPING............ True
VAL_SPLIT..................... 0.1
WEIGHT_DECAY.................. 0.0005
======================================================================

Using device: cpu

Loading CIFAR-100 data...
Files already downloaded and verified
Files already downloaded and verified
Training samples: 45000
Validation samples: 5000
Test samples: 10000

Creating CNN model...
Total parameters: 2,215,940

======================================================================
Starting Training
======================================================================
Epoch [  1/20] | Train Loss: 3.4686 | Train Acc: 17.78% | Val Loss: 3.1703 | Val Acc: 22.18%
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Epoch [  5/20] | Train Loss: 2.2285 | Train Acc: 41.32% | Val Loss: 2.6355 | Val Acc: 34.26%
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Epoch [ 10/20] | Train Loss: 1.8953 | Train Acc: 48.81% | Val Loss: 2.1146 | Val Acc: 44.00%
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Epoch [ 15/20] | Train Loss: 1.7262 | Train Acc: 52.85% | Val Loss: 2.0534 | Val Acc: 45.78%
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Epoch [ 20/20] | Train Loss: 1.6187 | Train Acc: 55.43% | Val Loss: 1.9228 | Val Acc: 47.92%

======================================================================
Training Completed
======================================================================
Best Validation Accuracy: 48.16%

Generating training history plots...
Training history plot saved to ./checkpoints/training_history_cnn.png
2026-01-10 19:20:16.686 python[84983:16952827] The class 'NSSavePanel' overrides the method ide
ntifier.  This method is implemented by class 'NSWindow'

Evaluating on test set...
Test Loss: 1.6476 | Test Accuracy: 55.21%

======================================================================
All Done!
======================================================================
```


CNN: Training and Validation: Loss and Accuracy
<img width="4168" height="1468" alt="training_history_cnn" src="https://github.com/user-attachments/assets/2c4584c8-b426-45db-82b4-0b45eb9c9572" />



## Final Model Checkpoint
The best performing model (Experiment 2.1) has been saved and can be loaded for verification.

* **Checkpoint File:** [`checkpoints/best_model_cnn.pth`](../checkpoints/best_model_cnn.pth)
* **Architecture:** Standard CNN (Adapted for CIFAR-100)
* **Performance:** 55.21% Accuracy

### How to Reproduce Results
To verify the performance metrics without retraining, run the evaluation script:

```bash
python evaluate_model.py
```


### Sample Evaluation
```text
Evaluating on device: cpu
Loading test data...
Files already downloaded and verified
Files already downloaded and verified
Test samples: 10000
Initializing CNN model...
Loading weights from ../checkpoints/best_model_cnn.pth...
Loaded checkpoint from Epoch 18
Running evaluation on Test Set...

========================================
FINAL RESULT: Task 2 (CIFAR-100)
========================================
Test Accuracy: 55.47%
Test Loss:     1.6249
========================================

```



### Download the Best Model from Checkpoint folder
* **Checkpoint File:** [`checkpoints/best_model_cnn.pth`](../checkpoints/best_model_cnn.pth)

