import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')  # 強制用互動視窗顯示圖 
import matplotlib.pyplot as plt
import numpy as np

# 設定隨機種子
torch.manual_seed(42)
np.random.seed(42)

# 資料生成與標準化
X, y = make_moons(n_samples=1000, noise=0.25)
X = StandardScaler().fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定義模型
class MoonNet(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self._activate(x)
        x = self.fc2(x)
        x = self._activate(x)
        x = self.fc3(x)
        return x

    def _activate(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        else:
            return x

# 訓練函數
def train(model, X, y, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# 決策邊界可視化
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid).argmax(dim=1).reshape(xx.shape)
    plt.contourf(xx, yy, preds, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

# 比較不同 activation
activations = ['none', 'relu', 'gelu']
plt.figure(figsize=(12, 4))

for i, act in enumerate(activations):
    print(f"--- Training with activation = {act.upper()} ---")
    net = MoonNet(activation=act)
    trained_net = train(net, X_train, y_train)
    plt.subplot(1, 3, i + 1)
    plot_decision_boundary(trained_net, X.numpy(), y.numpy(), title=f"Activation: {act.upper()}")

plt.tight_layout()
plt.show()
