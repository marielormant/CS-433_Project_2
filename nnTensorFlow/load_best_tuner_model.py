"""
Load best trial from Keras Tuner and train final model.
Loads saved hyperparameters and retrains on full training data.
"""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import keras_tuner as kt

# 1) Load and preprocess data (same as training)
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)

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

# Build labels from FI (not used as features!)
F = data[plies_cols].values
mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf
y = np.zeros(len(F), dtype=int)
max_FI = F_valid[has_failure].argmax(axis=1) + 1
y[has_failure] = max_FI

# Split data: 80/20 train/test, then 80/20 train/val on train
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

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

print(f"Test samples: {len(x_test):,}")

# 2) Load best trial hyperparameters from local trial_0071
import json
from tensorflow.keras import layers, regularizers

print("\n3) Loading best trial from trial_0071...")

# Load hyperparameters from build_config.json
with open("nnTensorFlow/trial_0071/build_config.json", "r") as f:
    config = json.load(f)

best_hps = config.get("config", {})

print(f"  Number of layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers', 0)):
    print(f"    Layer {i+1} units: {best_hps.get(f'units_layer_{i}')}")
    print(f"    Layer {i+1} dropout: {best_hps.get(f'dropout_layer_{i}')}")
print(f"  Batch normalization: {best_hps.get('batch_norm')}")
print(f"  L2 regularization: {best_hps.get('l2_reg')}")
print(f"  Learning rate: {best_hps.get('learning_rate')}")

# 3) Train final model with best hyperparameters

# Combine train and val for final training
x_train_full = np.vstack([x_train, x_val])
y_train_full = np.concatenate([y_train, y_val])

# Recompute class weights for combined set
class_weights_array_full = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_full),
    y=y_train_full
)
class_weights_dict_full = dict(enumerate(class_weights_array_full))

# Build model with best hyperparameters
best_model = keras.Sequential()
best_model.add(layers.Input(shape=(6,)))

num_layers = best_hps.get('num_layers', 2)
for i in range(num_layers):
    units = best_hps.get(f'units_layer_{i}', 256)
    l2_reg = best_hps.get('l2_reg', 0.001)
    
    best_model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg)))
    
    if best_hps.get('batch_norm', False):
        best_model.add(layers.BatchNormalization())
    
    best_model.add(layers.Activation('relu'))
    
    dropout = best_hps.get(f'dropout_layer_{i}', 0.3)
    best_model.add(layers.Dropout(dropout))

best_model.add(layers.Dense(17, activation='softmax'))

learning_rate = best_hps.get('learning_rate', 0.001)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
best_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


history = best_model.fit(
    x_train_full, y_train_full,
    epochs=100,
    batch_size=256,
    class_weight=class_weights_dict_full,
    validation_split=0.1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ],
    verbose=1
)

# 5) Evaluate on test set

y_test_pred = best_model.predict(x_test, batch_size=1024, verbose=0)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Overall metrics
accuracy = accuracy_score(y_test, y_test_pred_classes)
f1_macro = f1_score(y_test, y_test_pred_classes, average='macro')
f1_weighted = f1_score(y_test, y_test_pred_classes, average='weighted')

print("\nTest Results:")
print(f"  Accuracy:            {accuracy*100:.2f}%")
print(f"  F1 Score (macro):    {f1_macro*100:.2f}%")
print(f"  F1 Score (weighted): {f1_weighted*100:.2f}%")

# False negative analysis
is_actual_failure = (y_test > 0)
is_predicted_safe = (y_test_pred_classes == 0)
false_negatives = np.sum(is_actual_failure & is_predicted_safe)
total_failures = np.sum(is_actual_failure)
fn_rate = (false_negatives / total_failures * 100) if total_failures > 0 else 0

print(f"\nFalse Negatives:     {false_negatives:,} / {total_failures:,} ({fn_rate:.2f}%)")

# False positive analysis
is_actual_safe = (y_test == 0)
is_predicted_failure = (y_test_pred_classes > 0)
false_positives = np.sum(is_actual_safe & is_predicted_failure)
total_safe = np.sum(is_actual_safe)
fp_rate = (false_positives / total_safe * 100) if total_safe > 0 else 0

print(f"False Positives:     {false_positives:,} / {total_safe:,} ({fp_rate:.2f}%)")

# Per-class performance
print("\nPer-class performance:")
print(classification_report(y_test, y_test_pred_classes, zero_division=0, digits=3))

# Confusion matrix
print("\nConfusion matrix (first 6x6):")
cm = confusion_matrix(y_test, y_test_pred_classes)
print(cm[:6, :6])

# 6) Save final model and scaler
best_model.save('best_nn_model_tuned.h5')
with open('nn_scaler_tuned.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nSaved artifacts:")
print("  - best_nn_model_tuned.h5")
print("  - nn_scaler_tuned.pkl")

