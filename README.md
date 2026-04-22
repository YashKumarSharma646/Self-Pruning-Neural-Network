# 🧠 Self-Pruning Neural Network (Tredence AI Engineer Case Study)

## 📌 Overview
This project implements a self-pruning neural network where each weight is controlled by a learnable gate. The network dynamically learns which connections are important and removes unnecessary ones during training.

The goal is to demonstrate:
- Dynamic model pruning
- Sparsity vs accuracy trade-off
- Custom neural layer design

---

## ⚙️ Key Idea
Each weight is paired with a learnable parameter such that:
gate = sigmoid(gate_score)

The effective weight becomes:
pruned_weight = weight × gate

If gate → 0 → connection is pruned  
If gate → 1 → connection is retained  

---

## 🏗️ Architecture
- Input: 32×32×3 (CIFAR-10)
- Fully Connected Network:
  3072 → 256 → 128 → 10
- Activation: ReLU
- Custom Layer: PrunableLinear

---

## 📉 Loss Function
Total Loss = CrossEntropyLoss + λ × SparsityLoss

Where:
- SparsityLoss = sum of all gate values
- λ controls pruning strength

---

## 🧠 Why L1 Regularization Works
L1 regularization penalizes non-zero values linearly, pushing many gate values toward exactly zero. This results in a sparse network where only the most important connections remain active.

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 1e-5   | 44.47       | 1.10        |
| 1e-4   | 47.75       | 45.60       |
| 1e-3   | 44.28       | 98.10      |

---

## 📈 Observations
- Increasing λ increases sparsity.
- Moderate λ (1e-4) provides the best balance.
- High λ (1e-3) leads to aggressive pruning but slight accuracy drop.
- Model retains performance even after removing ~94% weights.

---

## 📊 Gate Distribution Analysis

### λ = 1e-5 (Minimal Pruning)
![λ=1e-5](results/gate_distribution1e5.png)
Gates remain spread across values → minimal pruning.

### λ = 1e-4 (Balanced Pruning)
![λ=1e-4](results/gate_distribution1e4.png)
Strong spike near zero with some active gates → optimal trade-off.

### λ = 1e-3 (Aggressive Pruning)
![λ=1e-3](results/gate_distribution1e3.png)
Most gates collapse to zero → highly sparse network.

---

## 🧪 Implementation Details
- Framework: PyTorch
- Dataset: CIFAR-10
- Optimizer: Adam
- Epochs: 5–10 (depending on λ)
- Device: GPU (Google Colab)

---

## ⚠️ Challenges Faced
- Loss scaling issue: sparsity loss (sum) initially dominated training and required tuning λ
- Attempted normalization using mean made sparsity too weak, so reverted to sum
- High λ required more epochs for pruning to take effect
- Notebook state issues caused incorrect results when model was not reset

---

## ✅ Key Learnings
- Importance of loss scaling in multi-objective optimization
- Practical behavior of L1 regularization
- Trade-off between model compression and accuracy
- Debugging deep learning training pipelines

---

## 🚀 How to Run
pip install -r requirements.txt

Then:
- Open self_pruning_model.ipynb
- Run all cells sequentially

---

## 📁 Project Structure
```
self-pruning-network/
│
├── self_pruning_model.ipynb
├── README.md
├── requirements.txt
│
├── results/
│   ├── gate_distribution1e3.png
│   ├── gate_distribution1e4.png
│   ├── gate_distribution1e5.png
```
---

## 🎯 Conclusion
This project successfully demonstrates that neural networks can learn to prune themselves dynamically. Significant compression (~94%) can be achieved while maintaining reasonable accuracy, showing the effectiveness of sparsity-driven optimization.

---

## 🙌 Author
Developed as part of Tredence AI Engineer Case Study.