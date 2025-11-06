import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)



# 建立一個簡單模型（只有一層 Linear）
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(3, 3)  # 輸入 3 維特徵，輸出 3 類分數

    def forward(self, x):
        logits = self.linear(x)              # Linear 計算分數（logits）
        probs = F.softmax(logits, dim=1)     # Softmax 將分數轉成機率
        return logits, probs

# 初始化模型
model = SimpleClassifier()
# 印出 Linear 權重對照表

feature_names = ["甜度", "酸度", "濕潤感"]
class_names = ["類別 0", "類別 1", "類別 2"]
print("\n📊 Linear 權重（每個特徵對每個類別的貢獻）\n")
weights = model.linear.weight.detach().numpy()  # 轉成 numpy 好處理

for class_idx, class_name in enumerate(class_names):
    print(f"🟢 {class_name}")
    for feat_idx, feat_name in enumerate(feature_names):
        w = weights[class_idx][feat_idx]
        print(f"   - {feat_name} 的權重：{w:.4f}")
    print()

# 假設輸入一筆甜點資料：[甜度, 酸度, 濕潤感]

input_data = torch.tensor([
    [100.0, 2.0, 1.0],
    [1.0, 0.0, 200.0]
])


# 模型推論
logits, probs = model(input_data)

# 預測類別（最大機率的 index）
predicted_class = torch.argmax(probs, dim=1)

# 顯示結果
for i, (x, p, pred) in enumerate(zip(input_data, probs, predicted_class)):
    print(f"🧾 第{i+1}筆資料")
    print("  🔢 輸入特徵：", x.tolist())
    print("  🎯 機率分布：", p.tolist())
    print("  ✅ 預測類別：", pred.item())
  
    print()
