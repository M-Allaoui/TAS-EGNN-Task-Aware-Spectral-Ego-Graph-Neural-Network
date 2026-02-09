#!/usr/bin/env python3
import os
import os.path as osp
import time
import argparse
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, APPNP
from torch_geometric.utils import (
    k_hop_subgraph, degree, to_undirected, remove_self_loops, coalesce
)

# Optional imports you already rely on (leave as-is for your project layout)
from utils import SomeUtils, Data2Pyg  # noqa: F401  (if unused in your run)
from datasets import define_dataset as define_fraud_dataset  # your CSV -> PyG helper

# ---------------------------
# Device & Repro
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# Pretty dataset stats (optional)
# ---------------------------
def print_dataset_stats(data: Data, name: str = "dataset"):
    num_nodes = data.num_nodes
    ei = data.edge_index
    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei, num_nodes=num_nodes)
    ei = coalesce(ei, num_nodes=num_nodes)
    num_edges_undirected = ei.size(1) // 2
    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1
    print(f"[{name}] nodes={num_nodes:,}  undirected_edges={num_edges_undirected:,}  "
          f"features={num_features:,}  classes={num_classes}")
    print(f"{name} & {num_nodes:,} & {num_edges_undirected:,} & {num_features:,} & {num_classes} \\\\")


def _ensure_num_classes(data: Data) -> int:
    return int(data.y.max().item()) + 1


def _make_one_class_masks(data: Data, normal_label: int = 0) -> Data:
    train_mask = data.train_mask.clone()
    val_mask   = data.val_mask.clone()
    test_mask  = data.test_mask.clone()
    normals = (data.y == normal_label)
    data.train_mask = train_mask & normals
    data.val_mask   = val_mask
    data.test_mask  = test_mask
    return data


# ---------------------------
# Dataset loading
# ---------------------------
def load_dataset(name: str, task: str = "supervised", normal_label: int = 0):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    n = name.lower()

    if n in ['cora', 'citeseer']:
        dataset = Planetoid(root=path, name=name, transform=NormalizeFeatures())
        data = dataset[0].to(device)
        return data, dataset.num_classes

    #Add your datasets here

    else:
        raise ValueError(f"Unsupported dataset: {name}")


# ---------------------------
# TAS-EGNN scoring & selection
# ---------------------------
@torch.no_grad()
def tas_egnn_greedy_fast(
    data: Data,
    model: nn.Module,
    ratio: float = 0.25,
    alpha: float = 0.1,
    beta: float  = 0.6,
    gamma: float = 0.4,
    radius: int  = 2
):
    model.eval()
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)
    entropy = -(probs * probs.log()).sum(dim=1)
    pred = probs.argmax(dim=1)
    misclass = (pred != data.y).float()

    train_nodes = data.train_mask.nonzero(as_tuple=True)[0]
    total_budget = int(ratio * train_nodes.size(0))
    scores = []

    for v in train_nodes:
        center = v.item()
        ego_nodes, ego_edge_index, _, _ = k_hop_subgraph(center, radius, data.edge_index, relabel_nodes=False)
        ego_deg = degree(ego_edge_index[0], num_nodes=data.num_nodes)
        var_deg = ego_deg[ego_nodes].var().item()
        score = alpha * var_deg + beta * entropy[center].item() + gamma * misclass[center].item()
        scores.append((center, score, ego_nodes))

    scores.sort(key=lambda x: x[1], reverse=True)
    selected_nodes = set()
    used_centers = set()
    covered = set()

    for center, _, ego_nodes in scores:
        new_nodes = set(ego_nodes.tolist()) - covered
        if not new_nodes:
            continue
        selected_nodes.update(ego_nodes.tolist())
        covered.update(ego_nodes.tolist())
        used_centers.add(center)
        if len(used_centers) >= total_budget:
            break

    coreset_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    if selected_nodes:
        coreset_mask[list(selected_nodes)] = True
    return coreset_mask


# ---------------------------
# GNN models
# ---------------------------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)
    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return self.prop(x, edge_index)


# ---------------------------
# Train / Eval
# ---------------------------
def train_eval(data: Data, model: nn.Module, mask: torch.Tensor, epochs=100, pos_label=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_acc, best_state = 0.0, None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = out.argmax(dim=1)
        val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    runtime = time.time() - start_time
    memory_used = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1).detach().cpu().numpy()

    test_mask_np = data.test_mask.cpu().numpy()
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = out.argmax(dim=1)[data.test_mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    C = probs.shape[1]
    if C == 2:
        pr_scores = probs[test_mask_np, int(pos_label)]
        pr_auc = average_precision_score(y_true, pr_scores)
    else:
        classes = list(range(C))
        Y_bin = label_binarize(y_true, classes=classes)
        pr_scores = probs[test_mask_np, :]
        pr_auc = average_precision_score(Y_bin, pr_scores, average='macro')

    return acc, f1, pr_auc, runtime, memory_used


# ---------------------------
# Pipeline
# ---------------------------
def run_pipeline(
    dataset_name: str,
    coreset_ratios,
    gnn_model: str = 'GraphSAGE',
    pre_epoch: int = 50,
    epochs: int = 500,
    num_runs: int = 3,
    alpha: float = 0.1,
    beta: float = 0.6,
    gamma: float = 0.4,
    radius: int = 2,
    task: str = 'oneclass',
    normal_label: int = 0,
    pos_label: int = 1
):
    results = []
    data, num_classes = load_dataset(dataset_name, task=task, normal_label=normal_label)

    model_map = {
        'GraphSAGE': GraphSAGE,
        'GCN': GCN,
        'GAT': GAT,
        'GIN': GIN,
        'APPNP': APPNPNet
    }
    if gnn_model not in model_map:
        raise ValueError(f"Unknown gnn_model '{gnn_model}'. Choose from {list(model_map.keys())}.")

    ModelClass = model_map[gnn_model]

    for ratio in coreset_ratios:
        accs, f1s, pr_aucs, runtimes, memories = [], [], [], [], []
        print(f"\n[RUN] Dataset={dataset_name} | Model={gnn_model} | Ratio={ratio}")
        for run in range(num_runs):
            print(f"  - Run {run + 1}/{num_runs}")

            # Warm-up on original train set
            model = ModelClass(data.num_features, 64, num_classes).to(device)
            _ = train_eval(data, model, data.train_mask, epochs=pre_epoch, pos_label=pos_label)

            # Coreset selection (ego-graph union budget via #centers)
            coreset_mask = tas_egnn_greedy_fast(
                data, model, ratio=ratio, alpha=alpha, beta=beta, gamma=gamma, radius=radius
            )

            # Re-train on coreset
            model = ModelClass(data.num_features, 64, num_classes).to(device)
            acc, f1, pr_auc, runtime, memory_used = train_eval(
                data, model, coreset_mask, epochs=epochs, pos_label=pos_label
            )

            accs.append(acc); f1s.append(f1); pr_aucs.append(pr_auc)
            runtimes.append(runtime); memories.append(memory_used)

            print(f"    Acc={acc:.4f}  F1={f1:.4f}  PR-AUC={pr_auc:.4f}  "
                  f"Time={runtime:.2f}s  Mem={memory_used:.2f}MB")

        results.append({
            'Dataset': dataset_name,
            'Model': gnn_model,
            'Coreset Ratio': ratio,
            'Accuracy Mean': mean(accs), 'Accuracy Std': (stdev(accs) if len(accs) > 1 else 0.0),
            'F1 Mean': mean(f1s), 'F1 Std': (stdev(f1s) if len(f1s) > 1 else 0.0),
            'PR-AUC Mean': mean(pr_aucs), 'PR-AUC Std': (stdev(pr_aucs) if len(pr_aucs) > 1 else 0.0),
            'Runtime Mean (s)': mean(runtimes), 'Runtime Std (s)': (stdev(runtimes) if len(runtimes) > 1 else 0.0),
            'Memory Mean (MB)': mean(memories), 'Memory Std (MB)': (stdev(memories) if len(memories) > 1 else 0.0),
        })

        print(f"[SUMMARY] Acc={mean(accs):.3f}±{(stdev(accs) if len(accs)>1 else 0):.3f}  "
              f"PR-AUC={mean(pr_aucs):.3f}±{(stdev(pr_aucs) if len(pr_aucs)>1 else 0):.3f}  "
              f"Time={mean(runtimes):.2f}s  Mem={mean(memories):.2f}MB")

    return pd.DataFrame(results)


# ---------------------------
# CLI helpers
# ---------------------------
def parse_ratios(r: str):
    # "0.5,0.25,0.15" -> [0.5, 0.25, 0.15]
    return [float(x.strip()) for x in r.split(',') if x.strip()]


def parse_datasets(d: str):
    # "cora,citeseer" -> ["cora","citeseer"]
    return [x.strip() for x in d.split(',') if x.strip()]


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="TAS-EGNN command-line runner")
    parser.add_argument('--datasets', type=str, default='cora',
                        help="Comma-separated dataset names (e.g., 'cora,citeseer,arxiv,flickr,banksim,paysim,ecc').")
    parser.add_argument('--model', type=str, default='GraphSAGE',
                        choices=['GraphSAGE', 'GCN', 'GAT', 'GIN', 'APPNP'])
    parser.add_argument('--ratios', type=str, default='0.5',
                        help="Comma-separated coreset ratios (e.g., '0.15,0.25,0.5' or '0.005,0.01,0.02').")
    parser.add_argument('--pre-epoch', type=int, default=50, help="Warm-up epochs before selection.")
    parser.add_argument('--epochs', type=int, default=500, help="Training epochs after selection.")
    parser.add_argument('--runs', type=int, default=3, help="Number of runs per ratio.")
    parser.add_argument('--alpha', type=float, default=0.1, help="Weight for structural (degree variance) term.")
    parser.add_argument('--beta', type=float, default=0.6, help="Weight for predictive entropy term.")
    parser.add_argument('--gamma', type=float, default=0.4, help="Weight for misclassification indicator.")
    parser.add_argument('--radius', type=int, default=2, help="Ego-graph hop radius.")
    parser.add_argument('--task', type=str, default='oneclass', choices=['supervised', 'oneclass'],
                        help="Training regime for fraud datasets.")
    parser.add_argument('--normal-label', type=int, default=0, help="Normal label for one-class training.")
    parser.add_argument('--pos-label', type=int, default=1, help="Positive class label for PR-AUC (binary).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--out', type=str, default=None, help="Path to save a combined CSV of results.")

    args = parser.parse_args()
    set_seed(args.seed)

    all_results = []
    datasets = parse_datasets(args.datasets)
    ratios = parse_ratios(args.ratios)

    for ds in datasets:
        df = run_pipeline(
            dataset_name=ds,
            coreset_ratios=ratios,
            gnn_model=args.model,
            pre_epoch=args.pre_epoch,
            epochs=args.epochs,
            num_runs=args.runs,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            radius=args.radius,
            task=args.task,
            normal_label=args.normal_label,
            pos_label=args.pos_label
        )
        all_results.append(df)

    if all_results:
        out_df = pd.concat(all_results, ignore_index=True)
        print("\n=== FINAL RESULTS ===")
        print(out_df.to_string(index=False))
        if args.out:
            os.makedirs(osp.dirname(args.out), exist_ok=True) if osp.dirname(args.out) else None
            out_df.to_csv(args.out, index=False)
            print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
