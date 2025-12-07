import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
print("Loading dataset...")
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(f"Dataset shape: {np.array(data).shape}")

# Extract epsilon features
eps_global = data['eps_global']
eps_df = pd.DataFrame(
    eps_global.tolist(), 
    columns=[f"eps{i}" for i in range(1, 7)]
)

# Extract plies features
plies_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]
plies_df = data[plies_cols]

# Combine features
data_df = pd.concat([eps_df, plies_df], axis=1)
print(f"Combined features shape: {data_df.shape}")

# Build labels
fi_cols = plies_cols  # Same as plies_cols
F = data_df[fi_cols].values  # (1_000_000, 16)

# Define label: 0 if no FI >= 1, otherwise argmax + 1
mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf

y = np.zeros(len(F), dtype=int)  # class 0 by default
max_FI = F_valid[has_failure].argmax(axis=1) + 1  # class 1..16
y[has_failure] = max_FI

print(f"Labels shape: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Class distribution
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))
print("\nClass distribution:")
for cls, count in class_counts.items():
    print(f"Class {cls}: {count} samples")

# Prepare features
x = data_df.values

# Split data: 80% train / 20% test with stratification
print("\nSplitting data...")
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {x_train.shape}")
print(f"Test set: {x_test.shape}")

# Further split training data for validation: 80% train / 20% validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Training set after validation split: {x_train.shape}")
print(f"Validation set: {x_val.shape}")

# Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Calculate class weights to handle imbalance
print("\nCalculating class weights...")
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_array))
print(f"Class weights: {class_weights_dict}")

# Build neural network model
print("\nBuilding neural network model...")

def build_model(input_dim, num_classes, learning_rate=0.001):
    """
    Build an optimized neural network for multi-class classification.
    
    Architecture:
    - Input layer: 22 features
    - Hidden layers with BatchNormalization and Dropout for regularization
    - Output layer: 17 classes (0-16)
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Second hidden layer
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Third hidden layer
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Fourth hidden layer
        layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with Adam optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Model parameters
input_dim = x_train.shape[1]  # 22 features
num_classes = len(np.unique(y))  # 17 classes

# Build model
model = build_model(input_dim, num_classes, learning_rate=0.001)
model.summary()

# Define callbacks for training optimization
callbacks = [
    # Early stopping: stop training if validation loss doesn't improve
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_nn_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
print("\nTraining neural network...")
batch_size = 256
epochs = 100

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# Load best model
print("\nLoading best model...")
model = keras.models.load_model('best_nn_model.h5')

# Evaluate on validation set
print("\n" + "="*50)
print("VALIDATION SET EVALUATION")
print("="*50)
y_val_pred = model.predict(x_val, batch_size=1024)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

val_accuracy = accuracy_score(y_val, y_val_pred_classes)
val_f1_macro = f1_score(y_val, y_val_pred_classes, average='macro')
val_f1_weighted = f1_score(y_val, y_val_pred_classes, average='weighted')

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1-Score (macro): {val_f1_macro:.4f}")
print(f"Validation F1-Score (weighted): {val_f1_weighted:.4f}")

# Evaluate on test set
print("\n" + "="*50)
print("TEST SET EVALUATION")
print("="*50)
y_test_pred = model.predict(x_test, batch_size=1024)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

test_accuracy = accuracy_score(y_test, y_test_pred_classes)
test_f1_macro = f1_score(y_test, y_test_pred_classes, average='macro')
test_f1_weighted = f1_score(y_test, y_test_pred_classes, average='weighted')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score (macro): {test_f1_macro:.4f}")
print(f"Test F1-Score (weighted): {test_f1_weighted:.4f}")

# Detailed classification report
print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("="*50)
print(classification_report(y_test, y_test_pred_classes, digits=4))

# Plot training history
print("\nPlotting training history...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss During Training')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Plot accuracy
axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Model Accuracy During Training')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training history saved to 'training_history.png'")

# Per-class performance analysis
print("\n" + "="*50)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("="*50)
for cls in range(num_classes):
    cls_mask = y_test == cls
    if cls_mask.sum() > 0:
        cls_pred = y_test_pred_classes[cls_mask]
        cls_true = y_test[cls_mask]
        cls_acc = accuracy_score(cls_true, cls_pred)
        print(f"Class {cls}: {cls_mask.sum()} samples, Accuracy: {cls_acc:.4f}")

print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
print(f"Best Model Performance:")
print(f"  - Test Accuracy: {test_accuracy:.4f}")
print(f"  - Test F1-Score (macro): {test_f1_macro:.4f}")
print(f"  - Test F1-Score (weighted): {test_f1_weighted:.4f}")
print(f"\nModel saved as: best_nn_model.h5")
print(f"Training history saved as: training_history.png")
print("="*50)
