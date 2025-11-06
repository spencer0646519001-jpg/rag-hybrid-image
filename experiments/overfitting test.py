import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 資料產生與標準化
x, y = make_moons(n_samples=100, noise=0.2, random_state=42)
x = StandardScaler().fit_transform(x)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# 模型定義（加了 Dropout，並降低 hidden size）
class OverfitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),           # Dropout 機率：30%
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = OverfitNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Early Stopping 設定
best_loss = float('inf')
patience = 10
trigger_times = 0

train_losses = []
val_losses = []

for epoch in range(1000):
    model.train()
    preds = model(x_train)
    loss = criterion(preds, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # 驗證 loss
    model.eval()
    with torch.no_grad():
        val_preds = model(x_test)
        val_loss = criterion(val_preds, y_test)
        val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

# 畫出 Loss 曲線
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve with Early Stopping")
plt.legend()
plt.grid()
plt.show()
