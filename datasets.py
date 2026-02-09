from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
#from imblearn.over_sampling import SMOTE
import os

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def to_pyg_graph(X, y, k=10):
    # Build a kNN graph
    knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(X)
    edge_index = knn.kneighbors_graph(X, mode='connectivity').tocoo()

    # Convert edge_index to PyTorch tensor
    edge_index = torch.tensor([edge_index.row, edge_index.col], dtype=torch.long)

    # Build PyG Data object
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.y = y
    data.num_classes = len(set(y.tolist()))

    # Optionally: create train/val/test split masks
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    idx_train, idx_rest, y_train, y_rest = train_test_split(idx, y, stratify=y, train_size=0.6, random_state=42)
    idx_val, idx_test, _, _ = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=0.5, random_state=42)

    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.train_mask[idx_train] = True
    data.val_mask = torch.zeros(len(y), dtype=torch.bool)
    data.val_mask[idx_val] = True
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask[idx_test] = True

    return data



def define_dataset(dataset="paysim", smote=False, scaler=False):

    class_names = ["Fraud", "Non Fraud"]
    if dataset=="ecc":
        # ---------------- ECC Load Dataset ----------------
        # NO SMOTE for ECC dataset
        # Size of dataset is (568630, 30) with equal number of samples in each class
        df = pd.read_csv("/home/students/mallaoui/PycharmProjects/datasets/ecc.csv")
        target = "Class"
        samples_per_class = {
            0: 10000,  # Class 0 → 1000 samples
            1: 10000,  # Class 1 → 200 samples
            # Add more classes if needed
        }

        # Subsample per class
        df = pd.concat([
            df[df[target] == cls].sample(n=n, random_state=42)
            for cls, n in samples_per_class.items()
        ])
        X = df.drop(columns=target)
        y = df[target]

    elif dataset == "paysim":
        # ---------------- Load and Preprocess PaySim ----------------
        # Works better with smote
        # Size of dataset: (6362620, 7), Original: Counter({0: 6354407, 1: 8213})
        df = pd.read_csv("/home/students/mallaoui/PycharmProjects/datasets/paysim.csv")
        target = "isFraud"   # change this to your actual label column name
        samples_per_class = {
            0: 10000,   # Class 0 → 1000 samples
            1: 8213,    # Class 1 → 200 samples
            # Add more classes if needed
        }

        # Subsample per class
        df = pd.concat([
            df[df[target] == cls].sample(n=n, random_state=42)
            for cls, n in samples_per_class.items()
        ])
        df.drop(columns=["nameOrig", "nameDest"], inplace=True)
        df['type'] = LabelEncoder().fit_transform(df['type'])
        y = df['isFraud']
        X = df.drop(columns=['isFraud', 'isFlaggedFraud'])

    elif dataset == "banksim":
        # ---------------- Banksim Load Dataset ----------------
        # Size of dataset: (594643, 7), Original: Counter({0: 587443, 1: 7200})

        df = pd.read_csv("/home/students/mallaoui/PycharmProjects/datasets/banksim.csv") #.sample(n=10000, random_state=42)
        target = "fraud"
        samples_per_class = {
            0: 10000,  # Class 0 → 1000 samples
            1: 7200,  # Class 1 → 200 samples
            # Add more classes if needed
        }

        # Subsample per class
        df = pd.concat([
            df[df[target] == cls].sample(n=n, random_state=42)
            for cls, n in samples_per_class.items()
        ])
        df = df.drop(['zipcodeOri','zipMerchant'],axis=1)
        col_categorical = df.select_dtypes(include= ['object']).columns
        for col in col_categorical:
            df[col] = df[col].astype('category')
        # categorical values ==> numeric values
        df[col_categorical] = df[col_categorical].apply(lambda x: x.cat.codes)
        X = df.drop(['fraud'],axis=1)
        y = df['fraud']

    # ---------------- Split Dataset ----------------
    print(np.shape(X))
    print(np.shape(y))
    print("Original:", Counter(y))


    """if scaler==True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if smote==True:
        smote = SMOTE(sampling_strategy={1: 1000}, random_state=42)  # Only generate 1,000 fraud samples
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(np.shape(X_train))
        print(np.shape(y_train))
        print("After SMOTE:", Counter(y_train))"""

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values (caused by conversion errors)
    X = X.dropna()

    # Align labels (drop corresponding labels for dropped rows)
    y = y[X.index]

    data = to_pyg_graph(X.values, y)
    return data