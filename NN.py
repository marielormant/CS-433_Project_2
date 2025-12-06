import pickle
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import NNOpti as opti


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



#class weighting not handle by sklearn MLPClassifier !!

#extract data into 80% train/ 20%test
# x = eps_df.values only epsilon ? 
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)


# NN classifier (MLP)
#number of neruons for each hidden layer : to tune
hidden1=opti.hidden1
hidden2=opti.hidden2
alpha = opti.alpha # L2 regularization parameter to tune
eta = opti.eta   # learning rate to tune 
batch_size = opti.batch_size #number of training samples used per gradient update )

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(hidden1, hidden2),  # 2 hidden layers, can tune
    learning_rate_init=eta,
    alpha=alpha,
    activation="relu", #function applied inside each neuron
    solver="adam",
    batch_size=batch_size, #mini-batch stochastic gradient descent
    max_iter=20,      # max number of full-passing over the entire training dataset
    verbose=True,
)
# Train with sample weights
mlp_clf.fit(x_train, y_train)


# Test
y_pred = mlp_clf.predict(x_test)

f1 = f1_score(y_test, y_pred, average="macro")
accuracy = accuracy_score(y_test, y_pred)

print("Macro F1:", f1)
print("Accuracy:", accuracy)