import numpy as np

# 假設我們的查詢向量 q
q = np.array([0.9, 0.1, 0.2])

# 三個文件的向量
docs = np.array([
    [0.8, 0.2, 0.3],
    [0.2, 0.9, 0.1],
    [-0.7, -0.1, -0.3]
])

# 歸一化（讓長度 = 1）
q_norm = q / np.linalg.norm(q)
docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)

# 計算每個文件與查詢的餘弦相似度
scores = docs_norm @ q_norm  # (3, 3) × (3,) → (3,)
print("各文件相似度：", scores)

# 取最高的前兩個
topk = np.argsort(-scores)[:2]
print("最相似的文件索引：", topk)
