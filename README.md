# TAS-EGNN-Task-Aware-Spectral-Ego-Graph-Neural-Network

TAS-EGNN is a **fast coreset selection** framework for Graph Neural Networks (GNNs).  
It scores nodes inside lightweight **ego-graphs** using three complementary signals:

1) **Local spectral complexity** (structural diversity)  
2) **Predictive uncertainty** (task difficulty)  
3) **Supervised error** (misclassification feedback)

A **greedy coverage** step then removes redundancy across selected ego-graphs.  
TAS-EGNN avoids global eigendecomposition and heavy bilevel optimization, delivering strong accuracy at small coreset ratios with **low runtime and memory**.

> **Status:** Paper and interface documented here. **Full code release coming soon.**

---

## ✨ Highlights
- **Task-aware** coreset selection with structural + predictive cues
- **Local** computations only (no full-graph spectral factorizations)
- Works across **citation**, **social**, **products**, and **transaction-fraud** graphs
- Reports **accuracy** (benchmarks) and **PR-AUC** (fraud), plus time & peak GPU memory

---

## 📦 Requirements

- Python ≥ 3.9  
- PyTorch (CUDA optional)  
- PyTorch Geometric (PyG)  
- Utilities: `numpy`, `scipy`, `pandas`, `scikit-learn`, `networkx`
