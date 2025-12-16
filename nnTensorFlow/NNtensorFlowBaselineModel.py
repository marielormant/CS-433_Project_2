"""
Neural Network Baseline Model (no hyperparameter optimization)
- Inputs: 6 epsilon features (standardized)
- Labels: 17 classes (0=no failure, 1-16 failure classes derived from FI)
- Split: 80/20 train/test
- Architecture: typical baseline (2 hidden layers, 256 units, dropout 0.3)
- Metrics: Accuracy, F1 (macro), FN rate, classification report
- Saves: nn_baseline_model.h5, nn_baseline_scaler.pkl
"""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import class_weight

np.random.seed(42)
tf.random.set_seed(42)

print("Neural Network Baseline Model — baseline (no tuning)")

# 1) Load and prepare data
with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(f"Samples: {len(data):,}")

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
F = data[plies_cols].values
mask_failure = (F >= 1.0)
has_failure = mask_failure.any(axis=1)
F_valid = F.copy()
F_valid[F_valid < 1.0] = -np.inf
y = np.zeros(len(F), dtype=int)
y[has_failure] = F_valid[has_failure].argmax(axis=1) + 1

print(f"Class distribution: {np.bincount(y)}")

# Split stratified 80/20
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Compute class weights for imbalanced data
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_array))

# 2) Baseline architecture (typical starting values)
baseline_architecture = {
    'num_layers': 2,
    'units_per_layer': 256,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100
}

print("\nBaseline hyperparameters:")
for k, v in baseline_architecture.items():
    print(f"  {k}: {v}")

# 3) Build baseline model
model = keras.Sequential([
    layers.InputLayer(input_shape=(6,)),
    
    # Hidden layer 1
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    
    # Hidden layer 2
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    
    # Output layer
    layers.Dense(17, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model.summary()

# 4) Train baseline model
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 5) Evaluate on test set
y_pred_probs = model.predict(x_test, batch_size=1024, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# False negatives analysis (missed failures)
is_actual_failure = (y_test > 0)
is_predicted_safe = (y_pred == 0)
false_negatives = np.sum(is_actual_failure & is_predicted_safe)
total_failures = np.sum(is_actual_failure)
fn_rate = false_negatives / total_failures if total_failures > 0 else 0

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"False Negatives: {false_negatives} / {total_failures}")
print(f"FN Rate: {fn_rate:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 6) Save model and scaler
model.save('nn_baseline_model.h5')
with open('nn_baseline_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nSaved: nn_baseline_model.h5")
print("Saved: nn_baseline_scaler.pkl")
print("Baseline training complete")
