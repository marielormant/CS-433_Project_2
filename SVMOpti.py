import pickle
import time
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


with open("dataset/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

data = pd.json_normalize(dataset)
print(np.array(data).shape) # (1 000 000, 17)

print(data.columns.tolist())
"""['eps_global', 'plies.0.0.FI_ft', 'plies.0.0.FI_fc', 
'plies.0.0.FI_mt', 'plies.0.0.FI_mc', 'plies.45.0.FI_ft', 
'plies.45.0.FI_fc', 'plies.45.0.FI_mt', 'plies.45.0.FI_mc',
 'plies.90.0.FI_ft', 'plies.90.0.FI_fc', 'plies.90.0.FI_mt',
 'plies.90.0.FI_mc', 'plies.-45.0.FI_ft', 'plies.-45.0.FI_fc', 
 'plies.-45.0.FI_mt', 'plies.-45.0.FI_mc']"""

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

x = data_df.values

# Build label for the all dataset

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

y = np.zeros(len(F), dtype=int)  # class 0 by default
max_FI = F_valid[has_failure].argmax(axis=1) + 1  # class 1..16
y[has_failure] = max_FI #replacing at failure row index the correspodning class


print(y[:10])
print(y.shape) #(1_000_000,)

#count of each class
numb_0 = len(np.where(y == 0)[0]) # no failure : 0 by default
# failure class 1..16 :
numb_1 = len(np.where(y == 1)[0]) 
numb_2 = len(np.where(y == 2)[0])
numb_3 = len(np.where(y == 3)[0])
numb_4 = len(np.where(y == 4)[0])
numb_5 = len(np.where(y == 5)[0])
numb_6 = len(np.where(y == 6)[0])
numb_7 = len(np.where(y == 7)[0])
numb_8 = len(np.where(y == 8)[0])
numb_9 = len(np.where(y == 9)[0])
numb_10 = len(np.where(y == 10)[0])
numb_11= len(np.where(y == 11)[0])
numb_12 = len(np.where(y == 12)[0])
numb_13 = len(np.where(y == 13)[0])
numb_14= len(np.where(y == 14)[0])
numb_15= len(np.where(y == 15)[0])
numb_16 = len(np.where(y == 16)[0])

print(numb_0)
print(numb_1)
print(numb_2)
print(numb_3)
print(numb_4)
print(numb_5)
print(numb_6)
print(numb_7)
print(numb_8)
print(numb_9)
print(numb_10)
print(numb_11)
print(numb_12)
print(numb_13)
print(numb_14)
print(numb_15)
print(numb_16)


# train one 100 000 first suffled rows 

#Randomly choose 100 000 rows from data_df without replacement
#Keep their original indices from the full dataset
#Return a new DataFrame with 100k rows
df_sub = data_df.sample(n=100_000, random_state=0)
y_sub  = y[df_sub.index]
x_sub  = df_sub.values

#extract data into 80% train/ 20%test
# x = eps_df.values only epsilon ? 

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0,
    stratify=y_sub
)

#scaling 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)


# SVM
#svm_clf = SVC(
    #kernel="rbf", #gaussian kernel
    #C=10.0, #penalty for misclassification
    #gamma="scale",
#)
#set of values to test during grid search
C_values = [0.12, 0.11, 0.1, 1, 10, 11, 12]
best_f1 = -np.inf
best_C = None
prevtime = 0

for C in C_values:
    #training
    #
    svm_clf1 = SVC(
        kernel="rbf", #gaussian kernel
        C=C, #penalty for misclassification
        gamma="scale",
    )
    svm_clf1.fit(x_train, y_train)

    #performance
    y_pred = svm_clf1.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="macro") # equally weighted average of classes
    accuracy = accuracy_score(y_test, y_pred)
    print(f"C={C}, SVC model, macro F1={f1:.4f}")
    print(f"C={C}, SVC model, accuracy={accuracy:.4f}")
    time = time.time()
    print(time - prevtime)
    prevtime = time

    if f1 > best_f1:
        best_f1 = f1
        best_C = C
    
    #second model
    svm_clf2 = LinearSVC(C=C, dual=False)  # dual=False is good when n_samples > n_features
    svm_clf2.fit(x_train, y_train)

    #performance
    y_pred2 = svm_clf1.predict(x_test)
    f12 = f1_score(y_test, y_pred2, average="macro") # equally weighted average of classes
    accuracy2 = accuracy_score(y_test, y_pred2)
    print(f"C={C}, linear SVC model, macro F1={f12:.4f}")
    print(f"C={C}, linear SVC model, accuracy={accuracy2:.4f}")
    time = time.time()
    print(time - prevtime)
    prevtime = time

    if f12 > best_f12:
        best_f12 = f12
        best_C2 = C



print("best C for SVC model : best_C ={best_C}")
print("best C for SVC linear model : best_C2={best_C2}")