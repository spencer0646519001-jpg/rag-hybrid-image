import numpy as np

# ---- 1️⃣ 規一化函數 ----
def normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

# ---- 2️⃣ 餘弦相似度 ----
def cosine_similarity(a, b):
    a = normalize(a)
    b = normalize(b)
    return np.dot(a, b)

# ---- 3️⃣ 搜尋函數 ----
def search(query_vec, docs, k=3):
    docs_norm = np.array([normalize(d) for d in docs])
    sims = np.dot(docs_norm, normalize(query_vec))
    topk_idx = np.argsort(-sims)[:k]
    return [(i, sims[i]) for i in topk_idx]

# ---- 4️⃣ 測試資料 ----
docs = [
    [1, 2, 3],
    [3, 2, 1],
    [0, 1, 1],
    [5, 0, 1],
]
query = [1, 1, 1]

# ---- 5️⃣ 執行搜尋 ----
results = search(query, docs, k=3)
for idx, score in results:
    print(f"Doc {idx} 相似度：{score:.3f}")
