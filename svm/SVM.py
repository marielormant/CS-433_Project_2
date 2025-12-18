import pickle
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# 1. Load dataset x

with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(np.array(data).shape) # (1 000 000, 17)

print(data.columns.tolist())

eps_global = data['eps_global']
print(np.array(eps_global).shape) # output (1 000 000,)

eps_df = pd.DataFrame(
    eps_global.tolist(), 
    columns=[f"eps{i}" for i in range(1, 7)]
)

print(eps_df.shape)  # output (1_000_000, 6)
print(eps_df.head())

plies_cols = ['plies.0.0.FI_ft', 'plies.0.0.FI_fc', 
'plies.0.0.FI_mt', 'plies.0.0.FI_mc', 'plies.45.0.FI_ft', 
'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
 'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt',
 'plies.90.0.FI_mc', 'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 
 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc']

plies_df = data[plies_cols]

data_df = pd.concat([eps_df, plies_df], axis=1)

print(data_df.shape)    # (1_000_000, 22)
print(data_df.head())


x = eps_df.values # Use only the first 6 features (eps1-eps6)

# 2. Build label for the all dataset

# List of the 16 FI feature columns (in a fixed order)
fi_cols = [
    'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 'plies.0.0.FI_mt', 'plies.0.0.FI_mc',
    'plies.45.0.FI_ft', 'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
    'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt', 'plies.90.0.FI_mc',
    'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc',
]

# Extract FI values, so extract the 16 FI columns and their corresponding rows
F = data_df[fi_cols].values  # (1_000_000, 16)

# Define label: 0 if no FI > 1, otherwise argmax + 1
mask_failure = (F >= 1) 
has_failure = mask_failure.any(axis=1)  #  (N,), True if at least one failure for each line

F_valid = F.copy()
F_valid[F_valid < 1] = -np.inf

y = np.zeros(len(F), dtype=int)  # class 0 by default, safe class 
max_FI = F_valid[has_failure].argmax(axis=1) + 1  # class 1..16
y[has_failure] = max_FI #replacing at failure row index the corresponding class


print(y[:10])
print(y.shape) #(1_000_000,)

# 3. Class weights to handle imbalanced dataset

# Count of each class
counts = {c: int((y == c).sum()) for c in range(17)}
print("Class counts:", counts)

N = data_df.shape[0]
K = data_df[fi_cols].shape[1] #number of failure classes

class_weights = {c: N / (K * n) for c, n in counts.items()}

class_weights = pd.Series(class_weights)
class_weights /= class_weights.mean() #normalize to have unit mean 
class_weights = class_weights.to_dict()

# 4. Extract data into 80% train/ 20% test

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

# 5. Standardize features : mean = 1; std = 0 and Polynomial Features degree = 2
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train = scaler.fit_transform(x_train)
x_train = poly.fit_transform(x_train)
x_test  = scaler.transform(x_test)
x_test  = poly.transform(x_test)

# 6. Define model : SVM and train with class weights of the 80 % train set
svm_clf = LinearSVC(C=1000, dual=False)  # C= 1 for baseline
sample_weight_train = np.array([class_weights[y_i] for y_i in y_train])
svm_clf.fit(x_train, y_train, sample_weight=sample_weight_train)

# 7. Test over the 20% test set
y_pred = svm_clf.predict(x_test)

# 8. Validation
f1 = f1_score(y_test, y_pred, average = "macro") # equally weighted average of classes
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)


print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro Precision: {precision:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Per-class detailed metrics
print(f"Detailed per-class metrics:")
print(f"{'Class':<6} {'Support':<10} {'FN':<8} {'FP':<8} {'Precision':<12} {'Recall':<10} {'F1':<10}")

results = []
for class_idx in range(17):
    # True Positives, False Positives, False Negatives
    TP = cm[class_idx, class_idx]
    FP = cm[:, class_idx].sum() - TP
    FN = cm[class_idx, :].sum() - TP
    
    # Support (number of true instances)
    support = (y_test == class_idx).sum()
    
    # Metrics : precision, recall, F1
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_class = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"{class_idx:<6} {support:<10} {FN:<8} {FP:<8} {prec:<12.4f} {rec:<10.4f} {f1_class:<10.4f}")
    
    results.append({
        'Class': class_idx,
        'Support': support,
        'False_Negatives': FN,
        'False_Positives': FP,
        'Precision': prec,
        'Recall': rec,
        'F1': f1_class
    })

# Summary statistics
df_results = pd.DataFrame(results)
print(f"Total False Negatives: {df_results['False_Negatives'].sum()}")
print(f"Total False Positives: {df_results['False_Positives'].sum()}")
print(f"Classes with recall < 0.3: {(df_results['Recall'] < 0.3).sum()}")
print(f"Classes with recall > 0.5: {(df_results['Recall'] > 0.5).sum()}")

