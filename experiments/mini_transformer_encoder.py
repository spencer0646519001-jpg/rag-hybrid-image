import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 固定亂數種子（讓每次初始化一致）
torch.manual_seed(42)

# 假設我們有 4 維特徵的序列，總共 5 個時間點（類似語音或句子中的字）
# 形狀：[序列長度, 批次大小, 特徵維度]
x = torch.randn(600, 1, 4)  

print("📥 輸入向量 x：")
print(x.squeeze(1))

# 定義一層 Transformer Encoder Layer（含 Multi-Head Attention）
encoder_layer = TransformerEncoderLayer(
    d_model=4,     # 每個向量維度（特徵數）
    nhead=2        # 注意力頭數（分成幾組看彼此）
)

# 組成完整的 Encoder 模組（這裡只用 1 層）
transformer_encoder = TransformerEncoder(encoder_layer, num_layers=3)

# 前向傳播
output = transformer_encoder(x)

print("\n📤 經過 Transformer Encoder 後的輸出向量：")
print(output.squeeze(1))
