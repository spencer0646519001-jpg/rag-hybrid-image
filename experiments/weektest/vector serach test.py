import numpy as np
from typing import List, Tuple

# ----- 工具函式 -----
def normalize(v: np.ndarray) -> np.ndarray:
    """把向量縮放成長度=1（若全零，直接回傳原向量避免除以零）"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n == 0, v, v / n)

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """兩個 1D 向量餘弦相似度"""
    u = normalize(u)
    v = normalize(v)
    return float(np.dot(u, v))

# ----- 模擬資料 -----
# 3個文件向量（例如嵌入後）
docs = np.array([
    [0.9, 0.1, 0.0],    # doc0：偏向 x1
    [0.7, 0.7, 0.0],    # doc1：介於 x1/x2 之間
    [0.0, 1.0, 0.0],    # doc2：純 x2 方向
], dtype=float)

# 查詢向量
q = np.array([1.0, 0.2, 0.0], dtype=float)

# ----- 計算並排序 -----
docs_norm = normalize(docs)        # (3,3)
q_norm = normalize(q)              # (3,)

# 與所有文件做點積（= 餘弦相似度，因為都已歸一化）
sims = docs_norm @ q_norm          # shape=(3,)
# 由大到小排序並取出索引
order = np.argsort(-sims)

print("cosine scores:", sims)
print("sorted doc indices (best → worst):", order)
for rank, i in enumerate(order, start=1):
    print(f"#{rank}: doc{i}, score={sims[i]:.3f}, vec={docs[i]}")



def search_cosine(query: np.ndarray, docs: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
    """
    回傳 [(doc_index, score)] 由高到低
    """
    qn = normalize(query)
    DN = normalize(docs)
    sims = DN @ qn          # (n_docs,)
    order = np.argsort(-sims)
    picks = order[:top_k]
    return [(int(i), float(sims[i])) for i in picks]

# 測一下
results = search_cosine(q, docs, top_k=3)
print(results)  # e.g. [(0, 0.982...), (1, 0.919...), (2, 0.196...)]

# 兩個查詢一次算：Q shape=(2,3)
Q = np.array([
    [1.0, 0.2, 0.0],   # q0
    [0.1, 1.0, 0.0],   # q1：靠近 x2
], dtype=float)

DN = normalize(docs)       # (3,3)
QN = normalize(Q)          # (2,3)

# 相似度矩陣：S = QN @ DN^T → shape=(2,3)，每列是1個查詢對所有文件
S = QN @ DN.T

print("similarity matrix (rows=query, cols=doc):\n", np.round(S, 3))

# 取每個查詢的 Top-K 索引
top_k = 2
best_idx = np.argsort(-S, axis=1)[:, :top_k]  # (2,2)
best_val = np.take_along_axis(S, best_idx, axis=1)

for qi in range(Q.shape[0]):
    print(f"\nQuery{qi}: {Q[qi]}")
    for rank in range(top_k):
        di = int(best_idx[qi, rank])
        sc = float(best_val[qi, rank])
        print(f"  #{rank+1}: doc{di}, score={sc:.3f}, vec={docs[di]}")

print('sim shape:', sims.shape)         # (n_queries, n_docs)
print('best_idx:\n', best_idx)
print('best_val:\n', best_val)
