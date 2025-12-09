"""
Load and evaluate the best model from interrupted Keras Tuner optimization
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

print("="*60)
print(" Loading Best Model from Keras Tuner")
print("="*60)

# -------------------------------------------------------
# 1. Load and preprocess data (same as training)
# -------------------------------------------------------
print("\n[1/5] Loading dataset...")
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)

eps_df = pd.DataFrame(
    data["eps_global"].tolist(),
    columns=[f"eps{i}" for i in range(1, 7)]
)

plies_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]
plies_df = data[plies_cols]

data_df = pd.concat([eps_df, plies_df], axis=1)
x = data_df.values

# Build labels
F = data_df[plies_cols].values
mask_failure = (F >= 1)
has_failure = mask_failure.any(axis=1)

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf
y = np.zeros(len(F), dtype=int)
max_FI = F_valid[has_failure].argmax(axis=1) + 1
y[has_failure] = max_FI

# Split data
print("\n[2/5] Splitting and scaling data...")
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

# Class weights
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_array))

print(f"Test samples: {len(x_test):,}")

# -------------------------------------------------------
# 2. Load the tuner and get best model
# -------------------------------------------------------
print("\n[3/5] Loading Keras Tuner results...")

# Recreate the tuner (it will load saved state)
from tensorflow.keras import layers, regularizers

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(22,)))
    
    num_layers = hp.Int('num_layers', min_value=2, max_value=4, step=1)
    
    for i in range(num_layers):
        if i == 0:
            units = hp.Int(f'units_layer_{i}', min_value=128, max_value=512, step=64)
        else:
            units = hp.Int(f'units_layer_{i}', min_value=32, max_value=256, step=32)
        
        l2_reg = hp.Float('l2_reg', min_value=0, max_value=0.01, step=0.001)
        
        model.add(layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg)))
        
        if hp.Boolean('batch_norm'):
            model.add(layers.BatchNormalization())
        
        model.add(layers.Activation('relu'))
        
        dropout = hp.Float(f'dropout_layer_{i}', min_value=0.1, max_value=0.5, step=0.1)
        model.add(layers.Dropout(dropout))
    
    model.add(layers.Dense(17, activation='softmax'))
    
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    hyperband_iterations=2,
    directory='nn_tuner_results',
    project_name='fn_minimization',
    overwrite=False  # Don't overwrite, load existing
)

print(f"Tuner loaded with {len(tuner.oracle.trials)} completed trials")

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n" + "="*60)
print("BEST HYPERPARAMETERS FROM {} TRIALS:".format(len(tuner.oracle.trials)))
print("="*60)
print(f"Number of layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"  Layer {i+1} units: {best_hps.get(f'units_layer_{i}')}")
    print(f"  Layer {i+1} dropout: {best_hps.get(f'dropout_layer_{i}')}")
print(f"Batch normalization: {best_hps.get('batch_norm')}")
print(f"L2 regularization: {best_hps.get('l2_reg')}")
print(f"Learning rate: {best_hps.get('learning_rate')}")
print("="*60)

# -------------------------------------------------------
# 3. Train final model with best hyperparameters
# -------------------------------------------------------
print("\n[4/5] Training final model with best hyperparameters...")
print("(Training on full train+val set for best performance)")

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

# Build and train final model
best_model = tuner.hypermodel.build(best_hps)

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

# -------------------------------------------------------
# 4. Evaluate on test set
# -------------------------------------------------------
print("\n[5/5] Evaluating on test set...")
print("="*60)
print("TEST SET EVALUATION")
print("="*60)

y_test_pred = best_model.predict(x_test, batch_size=1024, verbose=0)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Overall metrics
accuracy = accuracy_score(y_test, y_test_pred_classes)
f1_macro = f1_score(y_test, y_test_pred_classes, average='macro')
f1_weighted = f1_score(y_test, y_test_pred_classes, average='weighted')

print(f"\nOVERALL PERFORMANCE:")
print(f"  Accuracy:            {accuracy*100:.2f}%")
print(f"  F1 Score (macro):    {f1_macro*100:.2f}%")
print(f"  F1 Score (weighted): {f1_weighted*100:.2f}%")

# False negative analysis
print(f"\nFALSE NEGATIVE ANALYSIS:")
is_actual_failure = (y_test > 0)
is_predicted_safe = (y_test_pred_classes == 0)
false_negatives = np.sum(is_actual_failure & is_predicted_safe)
total_failures = np.sum(is_actual_failure)
fn_rate = (false_negatives / total_failures * 100) if total_failures > 0 else 0

print(f"  False Negatives:     {false_negatives:,} / {total_failures:,} ({fn_rate:.2f}%)")

# False positive analysis
is_actual_safe = (y_test == 0)
is_predicted_failure = (y_test_pred_classes > 0)
false_positives = np.sum(is_actual_safe & is_predicted_failure)
total_safe = np.sum(is_actual_safe)
fp_rate = (false_positives / total_safe * 100) if total_safe > 0 else 0

print(f"\nFALSE POSITIVE ANALYSIS:")
print(f"  False Positives:     {false_positives:,} / {total_safe:,} ({fp_rate:.2f}%)")

# Goal check
print("\n" + "="*60)
print("GOAL CHECK:")
print("="*60)
accuracy_goal = accuracy >= 0.90
f1_goal = f1_macro >= 0.90
fn_minimized = fn_rate < 5.0

print(f" Accuracy ≥ 90%:     {'YES ' if accuracy_goal else 'NO '} ({accuracy*100:.2f}%)")
print(f" F1 Score ≥ 90%:     {'YES ' if f1_goal else 'NO '} ({f1_macro*100:.2f}%)")
print(f" FN Rate < 5%:       {'YES ' if fn_minimized else 'NO '} ({fn_rate:.2f}%)")

if accuracy_goal and f1_goal and fn_minimized:
    print("\n ALL GOALS MET! ")
else:
    print("\n  Some goals not met.")

# Per-class performance
print("\n" + "="*60)
print("PER-CLASS PERFORMANCE:")
print("="*60)
print(classification_report(y_test, y_test_pred_classes, zero_division=0, digits=3))

# Confusion matrix
print("\nCONFUSION MATRIX (first 6x6):")
cm = confusion_matrix(y_test, y_test_pred_classes)
print(cm[:6, :6])

# -------------------------------------------------------
# 5. Save final model
# -------------------------------------------------------
print("\nSaving final model and scaler...")
best_model.save('best_nn_model_tuned.h5')

with open('nn_scaler_tuned.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Saved:")
print("  - best_nn_model_tuned.h5")
print("  - nn_scaler_tuned.pkl")

print("\n" + "="*60)
print(" EVALUATION COMPLETE!")
print("="*60)
