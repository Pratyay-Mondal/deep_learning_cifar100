# Task 1: Performance Report - CIFAR-10 dataset

### The report contains the training, evaluation after Stabilize the Existing Codebase.


### MLP results:
The MLP architecture has 2 hidden layers with batchnorm, relu and dropout. It achieved a Test Accuracy of 51.67%.

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
MODEL_TYPE.................... mlp
NUM_CLASSES................... 10
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

Creating MLP model...
Total parameters: 1,708,810

======================================================================
Starting Training
======================================================================
Epoch [  1/5] | Train Loss: 1.7027 | Train Acc: 39.26% | Val Loss: 1.5228 | Val Acc: 44.92%
Checkpoint saved to ./checkpoints/best_model_mlp.pth
Checkpoint saved to ./checkpoints/best_model_mlp.pth
Checkpoint saved to ./checkpoints/best_model_mlp.pth
Checkpoint saved to ./checkpoints/best_model_mlp.pth
Epoch [  5/5] | Train Loss: 1.4154 | Train Acc: 49.15% | Val Loss: 1.3711 | Val Acc: 51.08%
Checkpoint saved to ./checkpoints/best_model_mlp.pth

======================================================================
Training Completed
======================================================================
Best Validation Accuracy: 51.08%

Generating training history plots...
Training history plot saved to ./checkpoints/training_history_mlp.png
2026-01-20 12:02:24.549 python[59986:22309555] The class 'NSSavePanel' overrides the method identifier.  This method is implemented by class 'NSWindow'

Evaluating on test set...
Test Loss: 1.3617 | Test Accuracy: 51.67%

======================================================================
All Done!
======================================================================


```

MLP: Training and Validation: Loss and Accuracy
<img width="1400" height="500" alt="task1_MLP_Figure" src="https://github.com/user-attachments/assets/b07dcff0-13cf-4210-af42-8ffe01de253e" />


### CNN results:
The CNN architecture 2 blocks, each composed of Conv-BN-ReLU-Conv-BN-ReLU-MaxPool, followed by FC layers. It achieved a Test Accuracy of 76.04%.

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
NUM_CLASSES................... 10
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
Total parameters: 2,169,770

======================================================================
Starting Training
======================================================================
Epoch [  1/5] | Train Loss: 1.0582 | Train Acc: 62.42% | Val Loss: 0.8813 | Val Acc: 68.12%
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Checkpoint saved to ./checkpoints/best_model_cnn.pth
Epoch [  5/5] | Train Loss: 0.3714 | Train Acc: 87.20% | Val Loss: 0.6722 | Val Acc: 77.70%
Checkpoint saved to ./checkpoints/best_model_cnn.pth

======================================================================
Training Completed
======================================================================
Best Validation Accuracy: 77.70%

Generating training history plots...
Training history plot saved to ./checkpoints/training_history_cnn.png
2026-01-06 13:27:34.722 python[50035:14598844] The class 'NSSavePanel' overrides the method identifier.  This method is implemented by class 'NSWindow'

Evaluating on test set...
Test Loss: 0.7042 | Test Accuracy: 76.04%

======================================================================
All Done!
======================================================================


```


CNN: Training and Validation: Loss and Accuracy

<img width="1400" height="500" alt="task1_CNN_Figure_1" src="https://github.com/user-attachments/assets/bfb1cd71-b6b4-4057-989a-e8ae55fe7473" />
