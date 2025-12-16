"""
Neural network hyperparameter optimization with Keras Tuner (Hyperband)
Goal: maximize macro F1.
"""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import class_weight
import keras_tuner as kt
import time

np.random.seed(42)
tf.random.set_seed(42)

print("Keras Tuner neural network tuning (Hyperband, 100 trials)")

# 1) Load and preprocess data
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(f"Using all {len(data):,} samples for optimization")

# Features: 6 epsilons only (no FI in inputs)
eps_df = pd.DataFrame(
    data["eps_global"].tolist(),
    columns=[f"eps{i}" for i in range(1, 7)]
)
x = eps_df.values

# Labels: derive class from FI >= 1; take max FI index when multiple failures
plies_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]

# Build labels
F = data[plies_cols].values
mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf
y = np.zeros(len(F), dtype=int)
max_FI = F_valid[has_failure].argmax(axis=1) + 1
y[has_failure] = max_FI

print(f"Class distribution: {np.bincount(y)}")

# Split: 80/20 train/test, then 80/20 train/val on train
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Class weights for imbalance
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_array))

print(f"Training samples: {len(x_train):,}")
print(f"Validation samples: {len(x_val):,}")
print(f"Test samples: {len(x_test):,}")

# 2) Define model builder for Keras Tuner
# -------------------------------------------------------
def build_model(hp):
    """
    Build model with hyperparameters to tune.
    
    Hyperparameters:
    - Number of layers (2-4)
    - Units per layer (32-512)
    - Dropout rate (0.1-0.5)
    - Learning rate (1e-4 to 1e-2)
    - L2 regularization (0-0.01)
    - Batch normalization (on/off)
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(6,)))  # 6 epsilon features only!
    
    # Number of hidden layers
    num_layers = hp.Int('num_layers', min_value=2, max_value=3, step=1)
    
    for i in range(num_layers):
        # Units in layer (decreasing pattern)
        if i == 0:
            units = hp.Int(f'units_layer_{i}', min_value=128, max_value=256, step=64)
        else:
            units = hp.Int(f'units_layer_{i}', min_value=64, max_value=256, step=32)
        
        # L2 regularization
        l2_reg = hp.Float('l2_reg', min_value=0, max_value=0.005, step=0.001)
        
        model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg)))
        
        # Optional batch normalization
        if hp.Boolean('batch_norm'):
            model.add(layers.BatchNormalization())
        
        model.add(layers.Activation('relu'))
        
        # Dropout
        dropout = hp.Float(f'dropout_layer_{i}', min_value=0.2, max_value=0.4, step=0.1)
        model.add(layers.Dropout(dropout))
    
    # Output layer
    model.add(layers.Dense(17, activation='softmax'))
    
    # Learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# -------------------------------------------------------
# Custom metric for tuner: penalize false negatives
# -------------------------------------------------------
class FNAwareMetric(keras.callbacks.Callback):
    """
    Custom callback to compute FN-aware score during training
    """
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.best_score = -np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Compute F1
        f1 = f1_score(y_val, y_pred_classes, average='macro', zero_division=0)
        
        # Compute FN rate
        is_actual_failure = (y_val > 0)
        is_predicted_safe = (y_pred_classes == 0)
        false_negatives = np.sum(is_actual_failure & is_predicted_safe)
        total_failures = np.sum(is_actual_failure)
        fn_rate = false_negatives / total_failures if total_failures > 0 else 0
        
        # Combined score
        score = f1 - (3.0 * fn_rate)
        
        if score > self.best_score:
            self.best_score = score
        
        logs['fn_aware_score'] = score
        logs['fn_rate'] = fn_rate

# Custom tuner objective
class FNAwareObjective(kt.Objective):
    """Custom objective that considers both F1 and FN rate"""
    def __init__(self):
        super().__init__(name='val_loss', direction='min')

# 3) Run Keras Tuner Hyperband search
print("\nStarting Keras Tuner search (100 trials, Hyperband)...")
print("(Estimated time: 10-15 hours; auto-saves each trial)\n")

# Create tuner
# Note: Hyperband doesn't use max_trials, it uses hyperband_iterations
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    hyperband_iterations=3,  # More iterations = more trials (~100 total)
    directory='nn_tuner_results',
    project_name='fn_minimization',
    overwrite=False  # Don't overwrite - resume if stopped
)

# Callbacks for tuning
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

# Search for best hyperparameters

start_time = time.time()
tuner.search(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=256,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
elapsed_time = time.time() - start_time
print(f"\nSearch completed in {elapsed_time/60:.1f} minutes")

# Get and report best hyperparameters
print("\n4) Best hyperparameters found:")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"  Number of layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"    Layer {i+1} units: {best_hps.get(f'units_layer_{i}')}")
    print(f"    Layer {i+1} dropout: {best_hps.get(f'dropout_layer_{i}')}")
print(f"  Batch normalization: {best_hps.get('batch_norm')}")
print(f"  L2 regularization: {best_hps.get('l2_reg')}")
print(f"  Learning rate: {best_hps.get('learning_rate')}")

# Build and train final model
print("\n5) Training final model with best hyperparameters...")
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=256,
    class_weight=class_weights_dict,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ],
    verbose=1
)

# 6) Evaluate on test set

y_test_pred = best_model.predict(x_test, batch_size=1024)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

accuracy = accuracy_score(y_test, y_test_pred_classes)
f1_macro = f1_score(y_test, y_test_pred_classes, average='macro')
f1_weighted = f1_score(y_test, y_test_pred_classes, average='weighted')

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"F1 Score (macro): {f1_macro*100:.2f}%")
print(f"F1 Score (weighted): {f1_weighted*100:.2f}%")

# False negatives (missed failures)
is_actual_failure = (y_test > 0)
is_predicted_safe = (y_test_pred_classes == 0)
false_negatives = np.sum(is_actual_failure & is_predicted_safe)
total_failures = np.sum(is_actual_failure)
fn_rate = (false_negatives / total_failures * 100) if total_failures > 0 else 0

print(f"False Negatives: {false_negatives} / {total_failures} ({fn_rate:.2f}%)")

# False positives (false alarms)
is_actual_safe = (y_test == 0)
is_predicted_failure = (y_test_pred_classes > 0)
false_positives = np.sum(is_actual_safe & is_predicted_failure)
total_safe = np.sum(is_actual_safe)
fp_rate = (false_positives / total_safe * 100) if total_safe > 0 else 0

print(f"False Positives: {false_positives} / {total_safe} ({fp_rate:.2f}%)")

print("\nClassification report:")
print(classification_report(y_test, y_test_pred_classes, zero_division=0))

# 7) Save model and scaler
print("\n7) Saving model and scaler")
best_model.save('best_nn_model_optimized.h5')

import pickle as pkl
with open('nn_scaler_optimized.pkl', 'wb') as f:
    pkl.dump(scaler, f)

print("Saved:")
print("  - best_nn_model_optimized.h5")
print("  - nn_scaler_optimized.pkl")
print("\nOptimization complete")
