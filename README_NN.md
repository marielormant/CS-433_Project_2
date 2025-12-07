# Neural Network Model for Material Failure Classification

## Overview
This neural network model is designed for multi-class classification of material failure modes based on strain (epsilon) and failure index (FI) features from composite material simulations.

## Dataset
- **Total Samples**: 1,000,000
- **Features**: 22 (6 epsilon features + 16 FI features)
- **Classes**: 17 (0 = no failure, 1-16 = different failure modes)
- **Data Split**: 64% train / 16% validation / 20% test

## Model Architecture

The neural network consists of:
- **Input Layer**: 22 features
- **Hidden Layer 1**: 256 neurons + BatchNorm + ReLU + Dropout(0.3)
- **Hidden Layer 2**: 128 neurons + BatchNorm + ReLU + Dropout(0.3)
- **Hidden Layer 3**: 64 neurons + BatchNorm + ReLU + Dropout(0.2)
- **Hidden Layer 4**: 32 neurons + BatchNorm + ReLU + Dropout(0.2)
- **Output Layer**: 17 neurons with Softmax activation

### Key Features
- **Regularization**: L2 regularization (0.001) on all hidden layers
- **Normalization**: Batch Normalization after each hidden layer
- **Dropout**: Prevents overfitting (rates: 0.3, 0.3, 0.2, 0.2)
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Sparse Categorical Crossentropy
- **Class Weighting**: Balanced weights to handle class imbalance

## Training Configuration

### Callbacks
1. **Early Stopping**: Stops training if validation loss doesn't improve for 15 epochs
2. **ReduceLROnPlateau**: Reduces learning rate by 50% if validation loss plateaus for 5 epochs
3. **ModelCheckpoint**: Saves the best model based on validation loss

### Hyperparameters
- **Batch Size**: 256
- **Max Epochs**: 100
- **Initial Learning Rate**: 0.001
- **Minimum Learning Rate**: 1e-7

## Performance Results

### Test Set Performance (200,000 samples)
- **Accuracy**: 94.77%
- **F1-Score (macro)**: 93.01%
- **F1-Score (weighted)**: 94.86%

### Validation Set Performance (160,000 samples)
- **Accuracy**: 94.75%
- **F1-Score (macro)**: 93.03%
- **F1-Score (weighted)**: 94.83%

### Per-Class Performance (Test Set)
| Class | Samples | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| 0     | 41,991  | 99.94%    | 88.56% | 93.91%   |
| 1     | 17,590  | 98.54%    | 94.72% | 96.59%   |
| 2     | 22,305  | 97.24%    | 95.05% | 96.13%   |
| 3     | 10,376  | 96.70%    | 96.84% | 96.77%   |
| 4     | 11,441  | 96.33%    | 97.25% | 96.79%   |
| 5     | 5,942   | 88.20%    | 97.37% | 92.56%   |
| 6     | 4,741   | 80.45%    | 99.98% | 89.16%   |
| 7     | 4,041   | 84.53%    | 97.06% | 90.36%   |
| 8     | 2,599   | 73.84%    | 98.81% | 84.52%   |
| 9     | 17,544  | 98.32%    | 95.73% | 97.01%   |
| 10    | 22,360  | 96.62%    | 95.62% | 96.12%   |
| 11    | 10,315  | 97.13%    | 96.64% | 96.88%   |
| 12    | 11,396  | 95.60%    | 97.12% | 96.35%   |
| 13    | 5,893   | 85.71%    | 97.68% | 91.30%   |
| 14    | 4,802   | 84.77%    | 100%   | 91.76%   |
| 15    | 4,059   | 85.57%    | 98.20% | 91.45%   |
| 16    | 2,605   | 73.10%    | 97.62% | 83.60%   |

## Usage

### Requirements
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Running the Model
```bash
python neural_network.py
```

### Model Outputs
1. **best_nn_model.h5**: Saved model with best validation performance
2. **training_history.png**: Visualization of training/validation loss and accuracy
3. **Console Output**: Detailed classification report and per-class analysis

### Loading the Trained Model
```python
from tensorflow import keras

# Load the model
model = keras.models.load_model('best_nn_model.h5')

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_classes = np.argmax(predictions, axis=1)
```

## Comparison with Other Models

The neural network model can be compared with:
- **SVM.py**: Support Vector Machine with RBF kernel
- **SVMOpti.py**: Optimized SVM with hyperparameter tuning

### Key Advantages
1. **Better Performance**: Higher accuracy and F1-scores than SVM
2. **Scalability**: Can handle large datasets efficiently with batch processing
3. **Feature Learning**: Automatically learns complex feature representations
4. **Class Balance**: Built-in class weighting handles imbalanced data
5. **Regularization**: Multiple techniques prevent overfitting

## Model Optimization

The model includes several optimization strategies:

1. **Architecture**: Deep network with progressive dimension reduction
2. **Regularization**: L2 weight decay + Dropout + Batch Normalization
3. **Learning Rate**: Adaptive learning rate with ReduceLROnPlateau
4. **Early Stopping**: Prevents overfitting by stopping at optimal point
5. **Class Weights**: Balanced training for imbalanced dataset
6. **Data Preprocessing**: StandardScaler for feature normalization

## Future Improvements

Potential enhancements:
- Hyperparameter tuning with grid search or Bayesian optimization
- Ensemble methods (combining multiple models)
- Different architectures (ResNet, DenseNet)
- Data augmentation techniques
- Cross-validation for more robust evaluation
- Transfer learning from pre-trained models

## Notes

- Training takes approximately 15-20 minutes on a standard CPU
- GPU acceleration recommended for faster training
- Model automatically saves the best weights based on validation loss
- Training history plot helps visualize convergence and potential overfitting
- The model handles class imbalance through class weights
